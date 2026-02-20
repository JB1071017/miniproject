// Mobile Camera Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const toggleCameraBtn = document.getElementById('toggleCamera');
    const toggleStreamBtn = document.getElementById('toggleStream');
    const calibrateBtn = document.getElementById('calibrateBtn');
    const startCalibrationBtn = document.getElementById('startCalibration');
    const skipCalibrationBtn = document.getElementById('skipCalibration');
    const calibrationOverlay = document.getElementById('calibrationOverlay');
    const calStepSpan = document.getElementById('calStep');
    const cornerNameSpan = document.getElementById('cornerName');
    const touchCountSpan = document.getElementById('touchCount');
    const fpsCounterSpan = document.getElementById('fpsCounter');
    const processingStatusSpan = document.getElementById('processingStatus');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const sensitivitySlider = document.getElementById('sensitivity');
    const thresholdSlider = document.getElementById('threshold');
    const sensitivityValueSpan = document.getElementById('sensitivityValue');
    const thresholdValueSpan = document.getElementById('thresholdValue');
    const resetBackgroundBtn = document.getElementById('resetBackground');
    
    // Variables
    let stream = null;
    let isStreaming = false;
    let currentFacingMode = 'environment';
    let socket = null;
    let frameCount = 0;
    let lastFpsUpdate = Date.now();
    let fps = 0;
    let calibrationStep = 0;
    let isCalibrating = false;
    let touchPoints = [];
    
    // Camera corner names for calibration
    const cornerNames = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
    
    // Initialize
    init();
    
    function init() {
        setupSocketIO();
        setupEventListeners();
        showCalibrationOverlay();
    }
    
    function setupSocketIO() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        socket = io.connect(`${protocol}//${host}`);
        
        socket.on('connect', function() {
            console.log('Connected to server');
            updateConnectionStatus(true);
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            updateConnectionStatus(false);
        });
        
        socket.on('connected', function(data) {
            console.log('Server connection confirmed:', data);
        });
    }
    
    function setupEventListeners() {
        // Camera toggle
        toggleCameraBtn.addEventListener('click', toggleCamera);
        
        // Stream toggle
        toggleStreamBtn.addEventListener('click', toggleStream);
        
        // Calibration
        startCalibrationBtn.addEventListener('click', startCalibration);
        calibrateBtn.addEventListener('click', captureCalibrationPoint);
        skipCalibrationBtn.addEventListener('click', skipCalibration);
        
        // Settings
        sensitivitySlider.addEventListener('input', updateSensitivity);
        thresholdSlider.addEventListener('input', updateThreshold);
        resetBackgroundBtn.addEventListener('click', resetBackground);
    }
    
    async function toggleCamera() {
        try {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            
            currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
            await startCamera();
        } catch (error) {
            console.error('Error switching camera:', error);
            alert('Error switching camera. Please ensure camera permissions are granted.');
        }
    }
    
    async function startCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            
            video.onloadedmetadata = function() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            };
            
            updateConnectionStatus(true);
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Cannot access camera. Please check permissions and try again.');
        }
    }
    
    function toggleStream() {
        if (!stream) {
            alert('Please enable camera first');
            return;
        }
        
        isStreaming = !isStreaming;
        
        if (isStreaming) {
            toggleStreamBtn.innerHTML = '<i class="fas fa-pause"></i> Stop Stream';
            toggleStreamBtn.classList.add('active');
            startFrameProcessing();
        } else {
            toggleStreamBtn.innerHTML = '<i class="fas fa-play"></i> Start Stream';
            toggleStreamBtn.classList.remove('active');
        }
    }
    
    function startFrameProcessing() {
        if (!isStreaming) return;
        
        processFrame();
        
        // Update FPS counter
        frameCount++;
        const now = Date.now();
        if (now - lastFpsUpdate >= 1000) {
            fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
            fpsCounterSpan.textContent = fps;
            frameCount = 0;
            lastFpsUpdate = now;
        }
        
        requestAnimationFrame(startFrameProcessing);
    }
    
    function processFrame() {
        if (!isStreaming || video.readyState !== video.HAVE_ENOUGH_DATA) return;
        
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data
        const imageData = canvas.toDataURL('image/jpeg', 0.5);
        
        // Send to server
        if (socket && socket.connected) {
            socket.emit('video_frame', {
                image: imageData,
                timestamp: Date.now(),
                width: canvas.width,
                height: canvas.height
            });
        }
        
        // Update processing status
        processingStatusSpan.textContent = 'Processing';
        processingStatusSpan.style.color = '#4CAF50';
        
        // Clear status after delay
        setTimeout(() => {
            processingStatusSpan.textContent = 'Active';
            processingStatusSpan.style.color = '#2196F3';
        }, 100);
    }
    
    function showCalibrationOverlay() {
        calibrationOverlay.style.display = 'flex';
        isCalibrating = false;
    }
    
    function hideCalibrationOverlay() {
        calibrationOverlay.style.display = 'none';
        isCalibrating = false;
    }
    
    function startCalibration() {
        calibrationStep = 0;
        isCalibrating = true;
        startCalibrationBtn.style.display = 'none';
        calibrateBtn.style.display = 'inline-block';
        updateCalibrationUI();
    }
    
    function updateCalibrationUI() {
        calStepSpan.textContent = calibrationStep + 1;
        cornerNameSpan.textContent = cornerNames[calibrationStep];
    }
    
    function captureCalibrationPoint() {
        if (!isCalibrating || !socket) return;
        
        // Get the center point of the camera view
        const point = {
            x: canvas.width / 2,
            y: canvas.height / 2
        };
        
        // Send calibration point to server
        fetch('/calibrate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                step: 'point',
                point: [point.x, point.y]
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'calibrated') {
                alert('Calibration completed successfully!');
                hideCalibrationOverlay();
            } else if (data.status === 'next') {
                calibrationStep++;
                updateCalibrationUI();
            }
        })
        .catch(error => {
            console.error('Calibration error:', error);
            alert('Calibration failed. Please try again.');
        });
    }
    
    function skipCalibration() {
        hideCalibrationOverlay();
        alert('Calibration skipped. Basic touch detection will be used.');
    }
    
    function updateSensitivity() {
        const value = sensitivitySlider.value;
        sensitivityValueSpan.textContent = value;
        
        if (socket) {
            socket.emit('update_settings', {
                sensitivity: parseInt(value)
            });
        }
    }
    
    function updateThreshold() {
        const value = thresholdSlider.value;
        thresholdValueSpan.textContent = value;
        
        if (socket) {
            socket.emit('update_settings', {
                threshold: parseInt(value)
            });
        }
    }
    
    function resetBackground() {
        if (socket) {
            socket.emit('reset_background');
            alert('Background model reset. Hold still for a few seconds.');
        }
    }
    
    function updateConnectionStatus(connected) {
        if (connected) {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected';
        } else {
            statusDot.className = 'status-dot disconnected';
            statusText.textContent = 'Disconnected';
        }
    }
    
    // Listen for touch updates from server
    if (socket) {
        socket.on('touch_update', function(data) {
            touchPoints = data.touches || [];
            touchCountSpan.textContent = touchPoints.length;
        });
    }
    
    // Start camera automatically
    startCamera();
});