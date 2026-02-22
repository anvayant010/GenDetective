# GenDetective
GenDetective is a multimodal AI-content detection system designed as a Chrome browser extension backed by an integrated FastAPI backend. It detects whether images, videos, or text are AI-generated using a hybrid forensic approach that combines:
- Statistical and signal based analysis
- ML based Classification
- CLIP semantic modeling  
- LLM (Gemini) assisted multimodal reasoning
This integration and combination improves detection robustness, confidence calibration, and interpretability compared to single-method detectors and models.


## Key Features
### Image Detection
- EXIF Metadata Inspection
- Frequency spectrum & noise analysis
- Gemini-assisted forensic reasoning using structured prompts

### Video Detection
- Temporal consistency analysis
- Face symmetry, jitter, and texture analysis
- Frame-level forensic feature extraction
- Gemini multimodal reasoning on video segments

### Text Detection
- Locally trained ML classifier
- Logistic regression for fast & interpretable inference
- No external API required for text detection

## System Architecture
<p align="center">
  <img src="assets/system_architecture.png" alt="GenDetective System Architecture" width="890">
</p>

## Tech Stack
### Frontend(Extension + Dashboard)
- HTML, CSS, Javascript
- Chrome Extension APIs

### Backend
- FastAPI - Rest API
- Python
- NumPy, SciPy - statistical analysis
- OpenCV - image & video forensics
- Transformers (CLIP)
- Joblib - ML model loading

### AI/ML
- Scikit-learn, Colab - model building and training
- Gemini API - multimodal reasoning
- CLIP - Vision-Language Model


## Setup & Installation
### 1. Clone the repository
```bash
git clone https://github.com/your-username/GenDetective.git
cd GenDetective
```
### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

**Create a .env file**
```env
GEMINI_API_KEY=your_api_key_here
```
Get your personal Gemini API key from Google AI Studio

***First load the Extension by following the steps below then run the backend**

**Run the backend**
```bash
python backend.py
```
**Backend runs on**
```cpp
http://127.0.0.1:8000
```

### 3. Load Chrome Extension
1. Open `chrome://extensions`
2. Enable Dveloper Mode
3. Click Load Unpacked
4. Select the GenDetective folder

### The extension is now ready to use 🎉

### 4. Load Web Dashboard
GenDetective includes a modern web dashboard connected to the same backend. This performs all the function of that of the extension just in the web format.
1. Run the `index.html` file in the dashboard folder
2. The dashboard will run on personal localhost
   

## Extension Interface
<p align="center">
  <img src="assets/user_interface.png" alt="GenDetective Chrome Extension" width="450">
</p>

## Use Cases
- Fake news & misinformation detection
- Deepfake awareness tools
- Academic research
- AI safety & trust systems
- Browser-level content verification



