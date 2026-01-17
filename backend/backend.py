import os
import io
import sys
import time
import base64
import tempfile
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from PIL import Image
from scipy.stats import entropy

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from google import genai
from google.genai import types as gtypes

from joblib import load as joblib_load
import types as pytypes

# CONFIG & INIT

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

GEMINI_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=API_KEY)

app = FastAPI(title="GenDetective Backend (Balanced)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SCHEMAS

class ImageRequest(BaseModel):
    data: str
    mimeType: str = "image/png"

class VideoRequest(BaseModel):
    data: Optional[str] = None
    url: Optional[str] = None
    mimeType: str = "video/mp4"

class TextRequest(BaseModel):
    content: str

class DetectionResult(BaseModel):
    classification: str = Field(
        description="Label such as 'AI Generated', 'Likely AI Generated', 'Likely Real', 'Inconclusive'."
    )
    confidenceScore: int = Field(ge=0, le=100)
    justification: str
    forensicFactors: List[str] = Field(default_factory=list)
    geminiOpinion: Optional[Dict[str, Any]] = None
    heuristicScore: Optional[float] = None


def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return 0.0

def normalize01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))

# ENHANCED IMAGE FORENSICS (same features, milder interpretation)

def analyze_frequency_spectrum(gray_arr: np.ndarray) -> Dict[str, float]:
    f = np.fft.fft2(gray_arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    low = magnitude[cy - h//8: cy + h//8, cx - w//8: cx + w//8]
    low_energy = float(np.mean(low))

    mask = np.ones_like(magnitude)
    mask[cy - h//8: cy + h//8, cx - w//8: cx + w//8] = 0
    high_energy = float(np.mean(magnitude * mask))

    ratio = safe_div(low_energy, high_energy + 1e-9)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    radial_mean = np.bincount(r.ravel(), magnitude.ravel()) / (np.bincount(r.ravel()) + 1e-9)

    valid_idx = np.where(radial_mean > 0)[0]
    if len(valid_idx) > 10:
        idx = valid_idx[:min(50, len(valid_idx))]
        decay_rate = -np.polyfit(idx, np.log(radial_mean[idx] + 1e-9), 1)[0]
    else:
        decay_rate = 0.0

    return {
        "low_freq_energy": float(low_energy),
        "high_freq_energy": float(high_energy),
        "freq_ratio": float(ratio),
        "freq_decay_rate": float(decay_rate),
    }

def detect_upscaling_artifacts(pil_image: Image.Image) -> Dict[str, Any]:
    w, h = pil_image.size
    ar = w / h if h > 0 else 0
    suspicious_ars = [1.0, 1.5, 0.667, 1.778, 0.563]
    ar_susp = min(abs(ar - s) for s in suspicious_ars)

    power_of_2_dims = any([
        w in [256, 512, 768, 1024, 1536, 2048],
        h in [256, 512, 768, 1024, 1536, 2048],
    ])

    return {
        "aspect_ratio_suspicion": float(ar_susp),
        "power_of_2_dimensions": power_of_2_dims,
        "exact_dimension": f"{w}x{h}",
    }

def analyze_color_distribution(arr: np.ndarray) -> Dict[str, float]:
    if arr.ndim != 3:
        return {}

    feats = {}
    for i, ch in enumerate(["r", "g", "b"]):
        hist, _ = np.histogram(arr[:, :, i], bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-9)
        feats[f"{ch}_entropy"] = float(entropy(hist + 1e-9))
        peaks = hist[hist > hist.mean() + 2 * hist.std()]
        feats[f"{ch}_peaks"] = int(len(peaks))

    r = arr[:, :, 0].flatten()
    g = arr[:, :, 1].flatten()
    b = arr[:, :, 2].flatten()
    feats["rg_correlation"] = float(np.corrcoef(r, g)[0, 1])
    feats["rb_correlation"] = float(np.corrcoef(r, b)[0, 1])
    feats["gb_correlation"] = float(np.corrcoef(g, b)[0, 1])
    return feats

def detect_repeating_patterns(gray_arr: np.ndarray) -> float:
    h, w = gray_arr.shape
    if h < 100 or w < 100:
        return 0.0

    patch_size = 32
    patches = []
    for y in range(0, h - patch_size, patch_size * 2):
        for x in range(0, w - patch_size, patch_size * 2):
            p = gray_arr[y:y+patch_size, x:x+patch_size]
            if p.shape == (patch_size, patch_size):
                patches.append(p.flatten())
    if len(patches) < 4:
        return 0.0

    patches = np.array(patches)
    sims = []
    for i in range(len(patches)):
        for j in range(i + 1, min(i + 5, len(patches))):
            sims.append(np.corrcoef(patches[i], patches[j])[0, 1])
    return float(np.mean(sims)) if sims else 0.0

def analyze_edge_coherence(gray_arr: np.ndarray) -> Dict[str, float]:
    if not HAS_CV2:
        return {}
    img_u8 = (gray_arr * 255).astype("uint8")
    e1 = cv2.Canny(img_u8, 50, 150)
    e2 = cv2.Canny(img_u8, 100, 200)
    e3 = cv2.Canny(img_u8, 150, 250)

    d1 = float((e1 > 0).mean())
    d3 = float((e3 > 0).mean())
    consistency = safe_div(d3, d1 + 1e-9)

    if e2.sum() > 0:
        sobelx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(sobely, sobelx)
        hist, _ = np.histogram(orientation[e2 > 0], bins=36, range=(-np.pi, np.pi))
        o_entropy = float(entropy(hist + 1e-9))
    else:
        o_entropy = 0.0

    return {
        "edge_consistency": float(consistency),
        "edge_orientation_entropy": o_entropy,
        "edge_density_low": d1,
        "edge_density_high": d3,
    }

def extract_image_forensics_enhanced(pil_image: Image.Image) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    w, h = pil_image.size
    feats["dimensions"] = f"{w}x{h}"
    feats["aspect_ratio"] = round(safe_div(w, h), 3) if h else 0.0

    # EXIF
    exif = getattr(pil_image, "_getexif", lambda: None)()
    feats["has_exif"] = bool(exif)
    feats["camera_info"] = "unknown"
    feats["exif_completeness"] = 0.0
    if exif:
        try:
            from PIL.ExifTags import TAGS
            mapped = {TAGS.get(k, k): v for k, v in exif.items()}
            model = mapped.get("Model")
            make = mapped.get("Make")
            feats["camera_info"] = (f"{make or ''} {model or ''}").strip() or "present"
            important_tags = ["DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]
            present_count = sum(1 for tag in important_tags if tag in mapped)
            feats["exif_completeness"] = present_count / len(important_tags)
        except Exception:
            feats["camera_info"] = "present"

    # Arrays
    if pil_image.mode not in ("RGB", "RGBA"):
        rgb = pil_image.convert("RGB")
    else:
        rgb = pil_image
    arr = np.array(rgb).astype(np.uint8)
    gray = (
        cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        if HAS_CV2
        else np.array(rgb.convert("L")).astype(np.float32) / 255.0
    )

    feats["channel_means"] = list(np.round(arr.mean(axis=(0, 1)) / 255.0, 3))
    feats["channel_stds"] = list(np.round(arr.std(axis=(0, 1)) / 255.0, 3))
    feats["brightness_mean"] = float(np.round(gray.mean(), 3))
    feats["brightness_std"] = float(np.round(gray.std(), 3))

    feats.update(analyze_frequency_spectrum(gray))
    feats.update(detect_upscaling_artifacts(pil_image))
    feats.update(analyze_color_distribution(arr))
    feats["pattern_repetition"] = float(np.round(detect_repeating_patterns(gray), 3))
    feats.update(analyze_edge_coherence(gray))

    # Noise & sharpness 
    if HAS_CV2:
        lap = cv2.Laplacian((gray * 255).astype("uint8"), cv2.CV_64F)
        feats["sharpness_var"] = float(np.round(lap.var(), 2))
        blurred = cv2.GaussianBlur((gray * 255).astype("uint8"), (5, 5), 0)
        diff = (gray * 255).astype("float32") - blurred.astype("float32")
        feats["noise_std"] = float(np.round(np.std(diff), 2))
        feats["noise_mean"] = float(np.round(np.mean(np.abs(diff)), 2))
    else:
        feats["sharpness_var"] = 0.0
        feats["noise_std"] = 0.0
        feats["noise_mean"] = 0.0

    fmt = (pil_image.format or "").upper()
    feats["format"] = fmt or "unknown"
    return feats

def image_heuristic_score_enhanced(feats: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Much more conservative heuristic:
    - Scores typically between ~0 and 50.
    - Each cue is treated as weak / suggestive, NOT as proof.
    """
    score = 0.0
    factors: List[str] = []

    if not feats.get("has_exif"):
        score += 10
        factors.append("EXIF metadata missing (common in edited or web images; weak AI hint)")
    elif feats.get("exif_completeness", 0.0) < 0.3:
        score += 6
        factors.append("EXIF metadata present but partially incomplete")

    freq_ratio = feats.get("freq_ratio", 0.0)
    if freq_ratio > 100:
        score += 8
        factors.append("Unusual low/high frequency ratio (could suggest generative smoothing)")

    # Common AI output dimensions 
    if feats.get("power_of_2_dimensions"):
        score += 8
        factors.append("Resolution matches common model sizes (512/768/1024 etc., weak AI hint)")

    if feats.get("aspect_ratio_suspicion", 1.0) < 0.05:
        score += 5
        factors.append("Aspect ratio matches common AI presets")

    avg_entropy = np.mean([feats.get(f"{c}_entropy", 3.0) for c in ["r", "g", "b"]])
    if avg_entropy < 5.5:
        score += 8
        factors.append("Somewhat reduced color entropy (slightly limited palette)")

    corrs = [
        feats.get("rg_correlation", 0.0),
        feats.get("rb_correlation", 0.0),
        feats.get("gb_correlation", 0.0),
    ]
    if any(abs(c) > 0.95 for c in corrs):
        score += 6
        factors.append("Strong correlation between color channels (could be rendering artifact)")

    pattern_rep = feats.get("pattern_repetition", 0.0)
    if pattern_rep > 0.75:
        score += 10
        factors.append("Repeating local patterns detected (possible GAN-style artifact)")

    edge_consistency = feats.get("edge_consistency", 0.5)
    if edge_consistency < 0.2 or edge_consistency > 0.8:
        score += 6
        factors.append("Slightly unusual edge consistency across thresholds")

    noise_std = feats.get("noise_std", 0.0)
    if noise_std < 2.0:
        score += 8
        factors.append("Very low high-frequency noise (over-smoothing; could be AI or heavy denoising)")

    sharp = feats.get("sharpness_var", 0.0)
    if sharp < 50.0:
        score += 4
        factors.append("Lower sharpness variance than typical handheld photos")

    score = max(0.0, min(60.0, score))
    return score, factors

# ENHANCED VIDEO FORENSICS (also conservative)

def analyze_temporal_consistency(frames: List[np.ndarray]) -> Dict[str, float]:
    if len(frames) < 3:
        return {}
    flow_mags = []
    brightness_jumps = []
    for i in range(len(frames) - 1):
        bdiff = abs(frames[i+1].mean() - frames[i].mean())
        brightness_jumps.append(bdiff)
        if HAS_CV2:
            flow = cv2.calcOpticalFlowFarneback(
                (frames[i] * 255).astype("uint8"),
                (frames[i+1] * 255).astype("uint8"),
                None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_mags.append(mag.mean())
    bright_var = float(np.var(brightness_jumps)) if brightness_jumps else 0.0
    flow_var = float(np.var(flow_mags)) if flow_mags else 0.0
    return {
        "brightness_jump_variance": bright_var,
        "optical_flow_variance": flow_var,
        "avg_frame_difference": float(np.mean(brightness_jumps)) if brightness_jumps else 0.0,
    }

def analyze_face_artifacts(face_regions: List[np.ndarray]) -> Dict[str, float]:
    if not face_regions or not HAS_CV2:
        return {}
    sym_scores = []
    tex_vars = []
    edge_sharps = []
    for face in face_regions:
        if face.size == 0:
            continue
        h, w = face.shape[:2]
        if w > 10:
            left = face[:, : w//2]
            right = cv2.flip(face[:, w//2 :], 1)
            min_w = min(left.shape[1], right.shape[1])
            if min_w > 0:
                s = np.corrcoef(
                    left[:, :min_w].flatten(),
                    right[:, :min_w].flatten(),
                )[0, 1]
                sym_scores.append(float(s))
        tex_vars.append(float(face.var()))
        edges = cv2.Canny((face * 255).astype("uint8"), 100, 200)
        edge_sharps.append(float((edges > 0).mean()))
    result = {}
    if sym_scores:
        result["avg_face_symmetry"] = float(np.mean(sym_scores))
        result["face_symmetry_std"] = float(np.std(sym_scores))
    if tex_vars:
        result["avg_face_texture"] = float(np.mean(tex_vars))
    if edge_sharps:
        result["avg_face_edge_sharpness"] = float(np.mean(edge_sharps))
    return result

def extract_video_features_enhanced(path: str) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if not HAS_CV2:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        feats["has_cv2"] = False
        feats["file_size_mb"] = round(size_mb, 2)
        return feats

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        feats["has_cv2"] = False
        feats["note"] = "cannot open with cv2"
        return feats

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dur = frame_count / fps if fps > 0 else 0.0

    feats.update(
        {
            "has_cv2": True,
            "width": w,
            "height": h,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_sec": round(dur, 2),
        }
    )

    N = min(30, max(1, frame_count // max(1, int(fps)))) if frame_count > 0 else 0
    idxs = np.linspace(0, frame_count - 1, N, dtype=int) if N > 0 else []

    sampled_frames: List[np.ndarray] = []
    face_regions: List[np.ndarray] = []
    face_positions: List[List[Tuple[int,int,int,int]]] = []

    face_cascade = None
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except Exception:
        pass

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sampled_frames.append(gray)
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(
                (gray * 255).astype("uint8"),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            if len(faces) > 0:
                face_positions.append(list(faces))
                for (x, y, ww, hh) in faces:
                    fr = gray[y:y+hh, x:x+ww]
                    if fr.size > 0:
                        face_regions.append(fr)
            else:
                face_positions.append([])

    cap.release()

    feats.update(analyze_temporal_consistency(sampled_frames))
    feats.update(analyze_face_artifacts(face_regions))

    feats["sampled_frames"] = len(idxs)
    feats["detected_face_frames"] = sum(1 for fp in face_positions if len(fp) > 0)
    feats["face_detection_rate"] = safe_div(
        feats["detected_face_frames"], feats["sampled_frames"]
    )

    centroids = []
    for fp in face_positions:
        if len(fp) == 1:
            x, y, w0, h0 = fp[0]
            centroids.append((x + w0 / 2, y + h0 / 2))
    if len(centroids) > 1:
        centroids = np.array(centroids)
        feats["face_jitter"] = float(np.mean(np.std(centroids, axis=0)))
    else:
        feats["face_jitter"] = 0.0

    return feats

def video_heuristic_score_enhanced(feats: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    factors: List[str] = []

    if not feats.get("has_cv2"):
        if feats.get("file_size_mb", 0.0) < 1.0:
            score += 6
            factors.append("Very small file size (heavily compressed or short clip)")
        return score, factors

    bvar = feats.get("brightness_jump_variance", 0.0)
    if bvar > 0.001:
        score += 10
        factors.append("Brightness changes between frames show some instability")

    fvar = feats.get("optical_flow_variance", 0.0)
    if fvar > 50.0:
        score += 10
        factors.append("Optical flow variance is relatively high (motion inconsistencies)")

    face_sym = feats.get("avg_face_symmetry", 0.9)
    if face_sym < 0.75:
        score += 10
        factors.append("Face symmetry is somewhat irregular")

    face_tex = feats.get("avg_face_texture", 0.0)
    if face_tex < 100:
        score += 8
        factors.append("Facial texture appears smoother than typical camera footage")

    face_edge = feats.get("avg_face_edge_sharpness", 0.0)
    if face_edge < 0.02:
        score += 8
        factors.append("Face boundaries are slightly blurred compared to background")

    face_jitter = feats.get("face_jitter", 0.0)
    if feats.get("face_detection_rate", 0.0) > 0.3:
        if face_jitter < 1.0:
            score += 6
            factors.append("Face position is unusually stable across frames")
        elif face_jitter > 60.0:
            score += 8
            factors.append("Face position jitters noticeably across frames")

    if feats.get("face_detection_rate", 0.0) < 0.15 and feats.get("sampled_frames", 0) > 5:
        score += 6
        factors.append("Faces rarely detected across sampled frames")

    avg_diff = feats.get("avg_frame_difference", 0.0)
    if avg_diff < 0.005:
        score += 8
        factors.append("Very small frame-to-frame change (could be static or generated content)")

    score = max(0.0, min(60.0, score))
    return score, factors

# PROMPTS

def build_image_prompt_enhanced(feat_text: str, heuristic_score: float, factors: List[str]) -> Tuple[str, str]:
    system_instruction = (
        "You are a forensic AI image detection specialist. "
        "You should combine visual inspection with the provided technical metrics. "
        "Treat each heuristic feature as a weak signal: it can appear in both AI-generated and real images. "
        "Your goal is to give a balanced judgement about whether the image is AI-generated or real."
    )

    factors_text = "\n".join(f"- {f}" for f in factors) if factors else "- (No notable heuristic cues)"

    user_prompt = (
        "# IMAGE FORENSIC ANALYSIS\n\n"
        "## Technical Metrics\n"
        f"{feat_text}\n\n"
        "## Heuristic Summary (approximate, not definitive)\n"
        f"Heuristic AI-likelihood score (0–60 scale): {round(heuristic_score, 1)}\n"
        f"{factors_text}\n\n"
        "## Task\n"
        "Inspect the attached image carefully using both the visual content and these metrics.\n"
        "Return a JSON object with:\n"
        '  - \"classification\": one of [\"AI Generated\", \"Likely AI Generated\", '
        '\"Likely Real Photograph\", \"Real Photograph\", \"Inconclusive\"]\n'
        '  - \"confidenceScore\": integer 0–100\n'
        '  - \"justification\": 2–4 sentences describing key visual and forensic cues\n'
        '  - \"forensicFactors\": 3–6 short bullet-style strings summarizing evidence\n'
        "Use lower confidence or 'Inconclusive' when evidence is weak or mixed. "
        "Do not treat missing EXIF or common resolutions as strong proof on their own."
    )
    return system_instruction, user_prompt

def build_video_prompt_enhanced(stats_text: str, heuristic_score: float, factors: List[str]) -> Tuple[str, str]:
    system_instruction = (
        "You are a deepfake and AI video detection specialist. "
        "Use the provided temporal metrics and face statistics only as soft hints; "
        "they can appear in both genuine and synthetic videos. "
        "Focus on overall temporal coherence, facial consistency, and visual realism."
    )

    factors_text = "\n".join(f"- {f}" for f in factors) if factors else "- (No notable heuristic cues)"

    user_prompt = (
        "# VIDEO FORENSIC ANALYSIS\n\n"
        "## Technical Video Metrics\n"
        f"{stats_text}\n\n"
        "## Heuristic Summary (approximate, not definitive)\n"
        f"Heuristic deepfake-likelihood score (0–60 scale): {round(heuristic_score, 1)}\n"
        f"{factors_text}\n\n"
        "## Task\n"
        "Inspect the attached video, focusing on faces, motion, lighting, and consistency over time.\n"
        "Return a JSON object with:\n"
        '  - \"classification\": one of [\"AI Generated / Deepfake\", \"Likely AI Generated\", '
        '\"Likely Real Footage\", \"Real Footage\", \"Inconclusive\"]\n'
        '  - \"confidenceScore\": integer 0–100\n'
        '  - \"justification\": 2–4 sentences with concrete visual and temporal cues\n'
        '  - \"forensicFactors\": 3–6 short bullet-style strings summarizing evidence\n'
        "Use lower confidence or 'Inconclusive' if video quality is low or signals are contradictory."
    )
    return system_instruction, user_prompt

# ENSEMBLE

def ensemble_decision_enhanced(
    gemini_result: Dict[str, Any],
    heuristic_score: float,
    factors: List[str],
    is_video: bool = False,
) -> Dict[str, Any]:
    """
    Balanced ensemble:
    - Direction (AI vs Real) comes from Gemini's classification.
    - Heuristics slightly adjust confidence (20% weight).
    - Symmetric thresholds for AI vs Real to avoid AI bias.
    """
    gem_conf = float(gemini_result.get("confidenceScore", 50))
    gem_clf = str(gemini_result.get("classification", "Inconclusive"))
    lab = gem_clf.lower()

    heuristic_scaled = min(100.0, heuristic_score * (100.0 / 60.0)) if heuristic_score > 0 else 0.0

    # 80% Gemini, 20% heuristics
    final_conf = int(round(gem_conf * 0.8 + heuristic_scaled * 0.2))
    final_conf = max(0, min(100, final_conf))

    if "ai generated" in lab or "deepfake" in lab or "ai " in lab:
        if final_conf >= 75:
            final_label = "AI Generated" if not is_video else "AI Generated / Deepfake"
        elif final_conf >= 55:
            final_label = "Likely AI Generated"
        else:
            final_label = "Inconclusive"
    elif "real" in lab or "photograph" in lab or "footage" in lab:
        if final_conf >= 75:
            final_label = "Real Photograph" if not is_video else "Real Footage"
        elif final_conf >= 55:
            final_label = "Likely Real"
        else:
            final_label = "Inconclusive"
    else:
        final_label = "Inconclusive"

    return {
        "classification": final_label,
        "confidence": final_conf,
        "gemini_raw": gemini_result,
        "heuristic_score": round(heuristic_score, 1),
        "ensemble_weights": "Gemini: 80%, Heuristics: 20%",
    }

# GEMINI CALLER

def call_gemini_image_enhanced(image_bytes: bytes, mime_type: str) -> DetectionResult:
    pil_img = Image.open(io.BytesIO(image_bytes))
    feats = extract_image_forensics_enhanced(pil_img)
    heuristic_score, heuristic_factors = image_heuristic_score_enhanced(feats)

    feat_lines = [
        f"Dimensions: {feats.get('dimensions')} (AR: {feats.get('aspect_ratio')})",
        f"EXIF: {'Present' if feats.get('has_exif') else 'Missing'} "
        f"(Completeness: {feats.get('exif_completeness', 0)*100:.0f}%)",
        f"Frequency ratio (low/high): {feats.get('freq_ratio', 0):.2f}, decay: {feats.get('freq_decay_rate', 0):.3f}",
        f"Color entropy (avg RGB): {np.mean([feats.get(f'{c}_entropy', 0) for c in ['r','g','b']]):.2f}",
        f"Noise std: {feats.get('noise_std', 0):.2f}, sharpness var: {feats.get('sharpness_var', 0):.2f}",
        f"Pattern repetition score: {feats.get('pattern_repetition', 0):.3f}",
        f"Edge consistency: {feats.get('edge_consistency', 0):.3f}",
    ]
    feat_text = "\n".join(feat_lines)

    system_instruction, user_prompt = build_image_prompt_enhanced(
        feat_text, heuristic_score, heuristic_factors
    )

    image_part = gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, user_prompt],
            config=gtypes.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.10,
            ),
        )
        gem_json = DetectionResult.model_validate_json(resp.text).model_dump()
    except Exception as e:
        print(f"Gemini image call failed: {e}")
        gem_json = {
            "classification": "Inconclusive",
            "confidenceScore": 40,
            "justification": f"Analysis failed: {str(e)}",
            "forensicFactors": [],
        }

    ensemble = ensemble_decision_enhanced(
        gem_json, heuristic_score, heuristic_factors, is_video=False
    )

    justification = gem_json.get("justification") or "No justification provided"
    gem_factors = gem_json.get("forensicFactors") or []
    combined_factors = list(dict.fromkeys(gem_factors + heuristic_factors))[:8]

    final_justification = (
        f"{justification}\n\n"
        f"Ensemble weights: {ensemble.get('ensemble_weights', 'Gemini: 80%, Heuristics: 20%')}\n"
        f"Heuristic evidence (soft cues): "
        f"{', '.join(heuristic_factors[:3]) if heuristic_factors else 'No notable heuristic cues'}"
    )

    return DetectionResult(
        classification=ensemble["classification"],
        confidenceScore=ensemble["confidence"],
        justification=final_justification,
        forensicFactors=combined_factors,
        geminiOpinion=gem_json,
        heuristicScore=heuristic_score,
    )

def call_gemini_video_enhanced(video_content: Any, local_feats: Dict[str, Any]) -> DetectionResult:
    heuristic_score, heuristic_factors = video_heuristic_score_enhanced(local_feats)

    stats_lines = [
        f"Resolution: {local_feats.get('width')}x{local_feats.get('height')}, "
        f"FPS: {local_feats.get('fps')}, Duration: {local_feats.get('duration_sec')}s",
        f"Brightness jump variance: {local_feats.get('brightness_jump_variance', 0):.5f}",
        f"Optical flow variance: {local_feats.get('optical_flow_variance', 0):.2f}",
        f"Face detection rate: {local_feats.get('face_detection_rate', 0)*100:.1f}%",
        f"Face jitter: {local_feats.get('face_jitter', 0):.2f}",
        f"Avg face symmetry: {local_feats.get('avg_face_symmetry', 0):.3f}",
    ]
    stats_text = "\n".join(stats_lines)

    system_instruction, user_prompt = build_video_prompt_enhanced(
        stats_text, heuristic_score, heuristic_factors
    )

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[video_content, user_prompt],
            config=gtypes.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.12,
            ),
        )
        gem_json = DetectionResult.model_validate_json(resp.text).model_dump()
    except Exception as e:
        print(f"Gemini video call failed: {e}")
        gem_json = {
            "classification": "Inconclusive",
            "confidenceScore": 40,
            "justification": f"Analysis failed: {str(e)}",
            "forensicFactors": [],
        }

    ensemble = ensemble_decision_enhanced(
        gem_json, heuristic_score, heuristic_factors, is_video=True
    )

    justification = gem_json.get("justification") or "No justification provided"
    gem_factors = gem_json.get("forensicFactors") or []
    combined_factors = list(dict.fromkeys(gem_factors + heuristic_factors))[:8]

    final_justification = (
        f"{justification}\n\n"
        f"Ensemble weights: {ensemble.get('ensemble_weights', 'Gemini: 80%, Heuristics: 20%')}\n"
        f"Heuristic evidence (soft cues): "
        f"{', '.join(heuristic_factors[:3]) if heuristic_factors else 'No notable heuristic cues'}"
    )

    return DetectionResult(
        classification=ensemble["classification"],
        confidenceScore=ensemble["confidence"],
        justification=final_justification,
        forensicFactors=combined_factors,
        geminiOpinion=gem_json,
        heuristicScore=heuristic_score,
    )

# TEXT DETECTION

detector_mod = pytypes.ModuleType("detector")

def get_statistical_features(texts):
    import numpy as _np, re as _re
    feats = []
    for t in texts:
        words = t.split()
        wc = len(words) or 1
        unique_ratio = len(set(words)) / wc
        sentences = _re.split(r"[.!?]+", t)
        s_lens = [len(s.split()) for s in sentences if len(s.split()) > 5]
        sent_std = float(_np.std(s_lens)) if len(s_lens) > 1 else 0.0
        avg_len = float(_np.mean([len(w) for w in words])) if wc else 0.0
        feats.append([unique_ratio, sent_std, avg_len, 0.0, 0.0])
    return _np.array(feats)

detector_mod.get_statistical_features = get_statistical_features
sys.modules["detector"] = detector_mod

print("📌 Loading AI Text Detection Model...")
try:
    TEXT_MODEL = joblib_load("ai_detector_improved_model.joblib")
    print("✅ Text model loaded.")
except Exception as e:
    print(f"⚠️ Text model load failed: {e}")
    TEXT_MODEL = None

def classify_text_ai(content: str) -> DetectionResult:
    if TEXT_MODEL is None:
        raise RuntimeError("Text model not loaded")
    probs = TEXT_MODEL.predict_proba([content])[0]
    human_pct = float(probs[0] * 100)
    ai_pct = float(probs[1] * 100)
    confidence = int(round(max(human_pct, ai_pct)))

    if ai_pct > 80:
        label = "Very likely AI-generated text"
    elif ai_pct > 60:
        label = "Leaning AI-generated text"
    elif ai_pct > 40:
        label = "Inconclusive / mixed signals"
    elif ai_pct > 20:
        label = "Leaning human-written text"
    else:
        label = "Very likely human-written text"

    justification = (
        "The classifier uses TF-IDF character n-grams and stylometric features "
        "(sentence length variance, lexical diversity, and punctuation patterns) "
        "to estimate whether the text resembles AI-generated output or human writing."
    )
    factors = [
        f"AI score: {ai_pct:.1f}%",
        f"Human score: {human_pct:.1f}%",
    ]
    return DetectionResult(
        classification=label,
        confidenceScore=confidence,
        justification=justification,
        forensicFactors=factors,
        heuristicScore=None,
    )

# ROUTES

@app.post("/analyze_image", response_model=DetectionResult)
def analyze_image(req: ImageRequest):
    try:
        image_bytes = base64.b64decode(req.data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    try:
        return call_gemini_image_enhanced(image_bytes, req.mimeType)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze_video", response_model=DetectionResult)
def analyze_video(req: VideoRequest):
    if not req.data and not req.url:
        raise HTTPException(status_code=400, detail="Provide either 'data' or 'url'")

    temp_path = None
    try:
        # Local upload
        if req.data:
            raw = base64.b64decode(req.data)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(raw)
                temp_path = tmp.name

            local_feats = extract_video_features_enhanced(temp_path)

            uploaded = client.files.upload(file=temp_path)
            start = time.time()
            timeout = 90
            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", state)
            while state_name == "PROCESSING" and (time.time() - start) < timeout:
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
                state = getattr(uploaded, "state", None)
                state_name = getattr(state, "name", state)

            if state_name == "FAILED":
                raise RuntimeError("Gemini Files processing failed")

            return call_gemini_video_enhanced(uploaded, local_feats)

        # URL path
        else:
            local_feats = {"has_cv2": False, "note": "URL source; local stats unavailable"}
            video_part = gtypes.Part.from_uri(uri=req.url, mime_type=req.mimeType)
            return call_gemini_video_enhanced(video_part, local_feats)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/analyze_text", response_model=DetectionResult)
def analyze_text(req: TextRequest):
    if len(req.content.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please provide at least 20 characters")
    try:
        return classify_text_ai(req.content)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Text analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.get("/")
def root():
    return {
        "status": "GenDetective backend (Balanced) running",
        "note": "Heuristics softened; Gemini decides direction, heuristics only nudge confidence.",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
