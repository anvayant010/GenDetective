
const BACKEND = "http://127.0.0.1:8000";

let currentMode = "image";
let imageBase64 = null;
let imageMime = null;

let videoBase64 = null;
let videoMime = null;
let videoURL = "";

const $ = (id) => document.getElementById(id);
const resultBox = $("result");
const errorBox = $("error");
const analyzeBtn = $("analyze-btn");
const analyzeLabel = $("analyze-label");
const loaderWrap = $("loader");

function clearResult() {
  resultBox.style.display = "none";
  resultBox.innerHTML = "";
  errorBox.textContent = "";
}

function showError(msg) {
  errorBox.textContent = msg;
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  loaderWrap.style.display = isLoading ? "flex" : "none";
  analyzeLabel.textContent = isLoading ? "Analyzing…" : "Analyze";
}

function renderResult(data) {
  const classification = data.classification || "Result";
  const confidence = data.confidenceScore ?? "?";
  const justification = data.justification || "No explanation provided.";
  const factors = data.forensicFactors || [];

  let borderColor = "#10b981"; 
  if (classification.toLowerCase().includes("ai")) borderColor = "#ef4444";
  else if (classification.toLowerCase().includes("human")) borderColor = "#3b82f6";

  resultBox.style.display = "block";
  resultBox.style.borderLeftColor = borderColor;
  resultBox.innerHTML = `
    <div class="result-title">${classification}</div>
    <div class="result-line"><strong>Confidence:</strong> ${confidence}%</div>
    <div class="result-line"><strong>Summary:</strong> ${justification}</div>
    ${
      factors.length
        ? `<ul class="result-factors">
             ${factors.map((f) => `<li>${f}</li>`).join("")}
           </ul>`
        : ""
    }
  `;
}

// ----- MODE SWITCHING -----
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    if (!mode) return;

    currentMode = mode;
    clearResult();

    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    ["image", "video", "text"].forEach((m) => {
      const panel = document.getElementById(`panel-${m}`);
      if (panel) panel.classList.toggle("active", m === mode);
    });
  });
});

// ----- IMAGE HANDLING -----
$("image-upload-area").addEventListener("click", () => {
  $("image-input").click();
});

$("image-input").addEventListener("change", (e) => {
  clearResult();
  const file = e.target.files[0];
  if (!file) return;

  if (file.size > 20 * 1024 * 1024) {
    showError("Please choose an image smaller than 20MB.");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const dataURL = reader.result;
    imageBase64 = dataURL.split(",")[1];
    imageMime = file.type || "image/png";

    const imgEl = $("image-preview");
    imgEl.src = dataURL;
    imgEl.style.display = "block";
  };
  reader.readAsDataURL(file);
});

// ----- VIDEO HANDLING -----
$("video-upload-area").addEventListener("click", () => {
  $("video-input").click();
});

$("video-input").addEventListener("change", (e) => {
  clearResult();
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    const dataURL = reader.result;
    videoBase64 = dataURL.split(",")[1];
    videoMime = file.type || "video/mp4";
    videoURL = "";

    const videoEl = $("video-preview");
    videoEl.src = dataURL;
    videoEl.style.display = "block";
  };
  reader.readAsDataURL(file);
});

$("video-url").addEventListener("input", (e) => {
  clearResult();
  const url = e.target.value.trim();
  if (!url) {
    videoURL = "";
    return;
  }
  videoURL = url;
  videoBase64 = null;

  const videoEl = $("video-preview");
  videoEl.src = url;
  videoEl.style.display = "block";
});

// ----- ANALYZE BUTTON -----
analyzeBtn.addEventListener("click", async () => {
  clearResult();

  if (currentMode === "image") {
    await analyzeImage();
  } else if (currentMode === "video") {
    await analyzeVideo();
  } else if (currentMode === "text") {
    await analyzeText();
  }
});

// ----- ANALYSIS FUNCTIONS -----
async function analyzeImage() {
  if (!imageBase64 || !imageMime) {
    showError("Please upload an image first.");
    return;
  }

  setLoading(true);
  try {
    const resp = await fetch(`${BACKEND}/analyze_image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: imageBase64,
        mimeType: imageMime,
      }),
    });

    const json = await resp.json();
    if (!resp.ok || json.error) {
      showError(json.error || `Image analysis failed (${resp.status}).`);
    } else {
      renderResult(json);
    }
  } catch (err) {
    console.error(err);
    showError("Failed to reach backend. Is it running on port 8000?");
  } finally {
    setLoading(false);
  }
}

async function analyzeVideo() {
  if (!videoBase64 && !videoURL) {
    showError("Upload a video file or paste a video URL first.");
    return;
  }

  setLoading(true);
  try {
    const payload = videoBase64
      ? { data: videoBase64, mimeType: videoMime || "video/mp4" }
      : { url: videoURL, mimeType: "video/mp4" };

    const resp = await fetch(`${BACKEND}/analyze_video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await resp.json();
    if (!resp.ok || json.error) {
      showError(json.error || `Video analysis failed (${resp.status}).`);
    } else {
      renderResult(json);
    }
  } catch (err) {
    console.error(err);
    showError("Failed to reach backend. Is it running on port 8000?");
  } finally {
    setLoading(false);
  }
}

async function analyzeText() {
  const text = $("text-input").value.trim();
  if (text.length < 20) {
    showError("Please enter at least 20 characters of text.");
    return;
  }

  setLoading(true);
  try {
    const resp = await fetch(`${BACKEND}/analyze_text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: text }),
    });

    const json = await resp.json();
    if (!resp.ok || json.error) {
      showError(json.detail || json.error || `Text analysis failed (${resp.status}).`);
    } else {
      renderResult(json);
    }
  } catch (err) {
    console.error(err);
    showError("Failed to reach backend. Is it running on port 8000?");
  } finally {
    setLoading(false);
  }
}
