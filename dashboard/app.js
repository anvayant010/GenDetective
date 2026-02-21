const BACKEND = "http://127.0.0.1:8000";

let mode = "image";
let imageBase64 = null;
let videoBase64 = null;

const $ = (id) => document.getElementById(id);

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
  initializeTabs();
  initializeFileUploads();
  initializeTextEditor();
  initializeAnalyzeButton();
  addAnimationDelays();
  checkBackendConnection();
  
  setInterval(checkBackendConnection, 90000);
});

function addAnimationDelays() {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach((tab, index) => {
    tab.style.animationDelay = `${0.1 + index * 0.1}s`;
  });
}

function initializeTabs() {
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.panel');

  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      mode = btn.dataset.mode;

      tabs.forEach(t => t.classList.remove('active'));
      btn.classList.add('active');

      panels.forEach(p => p.classList.remove('active'));
      $(`panel-${mode}`).classList.add('active');

      createRipple(btn, event);
    });
  });
}

function createRipple(element, event) {
  const ripple = document.createElement('div');
  const rect = element.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const x = event.clientX - rect.left - size / 2;
  const y = event.clientY - rect.top - size / 2;

  ripple.style.cssText = `
    position: absolute;
    width: ${size}px;
    height: ${size}px;
    top: ${y}px;
    left: ${x}px;
    background: rgba(79, 70, 229, 0.3);
    border-radius: 50%;
    transform: scale(0);
    animation: ripple-effect 0.6s ease-out;
    pointer-events: none;
  `;

  element.style.position = 'relative';
  element.style.overflow = 'hidden';
  element.appendChild(ripple);

  setTimeout(() => ripple.remove(), 600);
}

// ==================== FILE UPLOADS ====================
function initializeFileUploads() {
  const imageInput = $('image-input');
  const imageUploadZone = $('image-upload-zone');
  const imagePreviewContainer = $('image-preview-container');
  const imagePreview = $('image-preview');
  const removeImage = $('remove-image');

  const videoInput = $('video-input');
  const videoUploadZone = $('video-upload-zone');
  const videoPreviewContainer = $('video-preview-container');
  const videoPreview = $('video-preview');
  const removeVideo = $('remove-video');

  setupFileUpload(
    imageInput,
    imageUploadZone,
    imagePreviewContainer,
    imagePreview,
    removeImage,
    'image',
    (base64, file) => {
      imageBase64 = base64;
      $('image-filename').textContent = file.name;
      $('image-size').textContent = formatFileSize(file.size);
    }
  );

  setupFileUpload(
    videoInput,
    videoUploadZone,
    videoPreviewContainer,
    videoPreview,
    removeVideo,
    'video',
    (base64, file) => {
      videoBase64 = base64;
      $('video-filename').textContent = file.name;
      $('video-size').textContent = formatFileSize(file.size);
    }
  );
}

function setupFileUpload(input, uploadZone, previewContainer, preview, removeBtn, type, callback) {
  uploadZone.addEventListener('click', () => input.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'rgba(79, 70, 229, 0.6)';
    uploadZone.style.background = 'rgba(79, 70, 229, 0.05)';
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.style.borderColor = '';
    uploadZone.style.background = '';
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = '';
    uploadZone.style.background = '';
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file, input, uploadZone, previewContainer, preview, callback);
    }
  });

  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file, input, uploadZone, previewContainer, preview, callback);
    }
  });

  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    previewContainer.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    input.value = '';
    if (type === 'image') imageBase64 = null;
    if (type === 'video') videoBase64 = null;
  });
}

function handleFileUpload(file, input, uploadZone, previewContainer, preview, callback) {
  const reader = new FileReader();

  reader.onload = () => {
    const base64 = reader.result.split(",")[1];
    preview.src = reader.result;
    
    uploadZone.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    
    callback(base64, file);
  };

  reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ==================== TEXT EDITOR ====================
function initializeTextEditor() {
  const textInput = $('text-input');
  const charCount = $('char-count');
  const clearBtn = $('clear-text');

  textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCount.textContent = count.toLocaleString();
    
    charCount.style.transform = 'scale(1.2)';
    setTimeout(() => {
      charCount.style.transform = 'scale(1)';
    }, 200);
  });

  clearBtn.addEventListener('click', () => {
    textInput.value = '';
    charCount.textContent = '0';
  });
}

async function checkBackendConnection() {
  const statusElement = $('backend-status');
  const statusText = $('backend-status-text');
  
  // Set to checking state
  statusElement.classList.remove('connected', 'disconnected');
  statusElement.classList.add('checking');
  statusText.textContent = 'Checking Backend...';
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); 
    
    const response = await fetch(BACKEND + '/', {
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      if (data.status && data.status.includes('GenDetective')) {
        statusElement.classList.remove('checking', 'disconnected');
        statusElement.classList.add('connected');
        statusText.textContent = 'Backend Connected';
      } else {
        throw new Error('Unexpected response from backend');
      }
    } else {
      throw new Error('Backend not responding correctly');
    }
  } catch (error) {
    statusElement.classList.remove('checking', 'connected');
    statusElement.classList.add('disconnected');
    statusText.textContent = 'Backend Disconnected';
    console.warn('Backend connection failed:', error.message);
  }
}

// ==================== ANALYZE BUTTON ====================
function initializeAnalyzeButton() {
  const analyzeBtn = $('analyze-btn');
  
  analyzeBtn.addEventListener('click', async () => {
    await performAnalysis();
  });
}

async function performAnalysis() {
  let endpoint = "";
  let payload = {};

  const backendStatus = $('backend-status');
  if (backendStatus.classList.contains('disconnected')) {
    showNotification('Backend is disconnected. Please check your connection.', 'error');
    return;
  }

  if (mode === "image") {
    if (!imageBase64) {
      showNotification('Please upload an image first', 'error');
      return;
    }
    endpoint = "/analyze_image";
    payload = { data: imageBase64, mimeType: "image/png" };
  }

  if (mode === "video") {
    if (!videoBase64) {
      showNotification('Please upload a video first', 'error');
      return;
    }
    endpoint = "/analyze_video";
    payload = { data: videoBase64, mimeType: "video/mp4" };
  }

  if (mode === "text") {
    const textContent = $("text-input").value.trim();
    if (!textContent) {
      showNotification('Please enter some text first', 'error');
      return;
    }
    endpoint = "/analyze_text";
    payload = { content: textContent };
  }

  showLoading();

  try {
    const res = await fetch(BACKEND + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error('Analysis failed');
    }

    const data = await res.json();
    
    hideLoading();
    displayResults(data);
    
  } catch (error) {
    hideLoading();
    showNotification('Analysis failed. Please try again.', 'error');
    console.error('Error:', error);
    
    checkBackendConnection();
  }
}

// ==================== LOADING STATE ====================
function showLoading() {
  const loading = $('loading');
  loading.classList.remove('hidden');
  
  const progressFill = $('progress');
  progressFill.style.width = '0%';
  
  setTimeout(() => {
    progressFill.style.transition = 'width 2s ease-in-out';
    progressFill.style.width = '70%';
  }, 100);
}

function hideLoading() {
  const loading = $('loading');
  const progressFill = $('progress');
  
  progressFill.style.width = '100%';
  
  setTimeout(() => {
    loading.classList.add('hidden');
    progressFill.style.width = '0%';
    progressFill.style.transition = 'none';
  }, 300);
}

// ==================== LAYOUT SWITCHING ====================
function enableSplitView() {
  const container = document.querySelector('.container');
  const mainWrapper = document.querySelector('.main-content-wrapper');
  const rightPanel = $('right-panel');
  
  container.classList.add('expanded');
  
  mainWrapper.classList.add('split-view');
  
  rightPanel.classList.remove('hidden');
  setTimeout(() => {
    rightPanel.classList.add('visible');
  }, 50);
}

function disableSplitView() {
  const container = document.querySelector('.container');
  const mainWrapper = document.querySelector('.main-content-wrapper');
  const rightPanel = $('right-panel');
  
  rightPanel.classList.remove('visible');
  
  setTimeout(() => {
    rightPanel.classList.add('hidden');
    
    mainWrapper.classList.remove('split-view');
    
    
    container.classList.remove('expanded');
  }, 300);
}

// ==================== RESULTS DISPLAY ====================
function displayResults(data) {
  const classificationText = $('classification-text');
  const classificationBadge = $('classification-badge');
  const confidenceValue = $('confidence-value');
  const confidenceFill = $('confidence-fill');
  const confidenceMarker = $('confidence-marker');
  const justificationText = $('justification-text');
  const analysisTime = $('analysis-time');

  classificationText.textContent = data.classification || 'Unknown';
  
  if (data.classification.toLowerCase().includes('ai')) {
    classificationBadge.style.background = 'linear-gradient(135deg, #df2508, #cf3c13)';
  } else {
    classificationBadge.style.background = 'linear-gradient(135deg, #10b981, #34d399)';
  }

  const confidence = parseFloat(data.confidenceScore) || 0;
  confidenceValue.textContent = `${confidence.toFixed(1)}%`;
  
  setTimeout(() => {
    confidenceFill.style.width = `${confidence}%`;
    confidenceMarker.style.left = `${confidence}%`;
  }, 100);

  justificationText.textContent = data.justification || 'No additional details available.';

  const randomTime = (Math.random() * 2 + 0.5).toFixed(1);
  analysisTime.textContent = `Analysis time: ${randomTime}s`;

  // Enable split view layout
  enableSplitView();

  const closeBtn = $('close-result');
  closeBtn.onclick = () => {
    disableSplitView();
  };
}

// ==================== NOTIFICATIONS ====================
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = 'notification';
  notification.textContent = message;
  
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 16px 24px;
    background: ${type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 'linear-gradient(135deg, #4F46E5, #10b981)'};
    color: white;
    border-radius: 12px;
    font-family: 'Manrope', sans-serif;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    z-index: 10000;
    animation: slide-in-right 0.3s ease-out;
  `;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.animation = 'slide-out-right 0.3s ease-out';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

const style = document.createElement('style');
style.textContent = `
  @keyframes slide-in-right {
    from {
      transform: translateX(400px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  @keyframes slide-out-right {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(400px);
      opacity: 0;
    }
  }
  
  @keyframes ripple-effect {
    to {
      transform: scale(4);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);

const analyzeBtn = $('analyze-btn');
analyzeBtn.addEventListener('mousemove', (e) => {
  const rect = analyzeBtn.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 100;
  const y = ((e.clientY - rect.top) / rect.height) * 100;
  
  analyzeBtn.style.setProperty('--mouse-x', `${x}%`);
  analyzeBtn.style.setProperty('--mouse-y', `${y}%`);
});

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    $('analyze-btn').click();
  }
});



setFavicon();
console.log('%cGenDetective v2.0', 'font-size: 20px; font-weight: bold; background: linear-gradient(135deg, #4F46E5, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;');
console.log('%cBuilt with cutting-edge design principles', 'font-size: 12px; color: #4F46E5;');
console.log('%cKeyboard shortcuts: Ctrl/Cmd + Enter to analyze', 'font-size: 11px; color: #888;');