// Projector Display Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const canvas = new fabric.Canvas('mainCanvas');
    const touchIndicators = document.getElementById('touchIndicators');
    const activeTouchesSpan = document.getElementById('activeTouches');
    const touchPositionsDiv = document.getElementById('touchPositions');
    const toolButtons = document.querySelectorAll('.tool-btn');
    const colorOptions = document.querySelectorAll('.color-option');
    const customColorInput = document.getElementById('customColor');
    const brushSizeSlider = document.getElementById('brushSize');
    const brushSizeValueSpan = document.getElementById('brushSizeValue');
    const clearCanvasBtn = document.getElementById('clearCanvas');
    const undoBtn = document.getElementById('undo');
    const redoBtn = document.getElementById('redo');
    const saveCanvasBtn = document.getElementById('saveCanvas');
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    const uploadPdfBtn = document.getElementById('uploadPdf');
    const uploadPresentationBtn = document.getElementById('uploadPresentation');
    const fileInput = document.getElementById('fileInput');
    const documentList = document.getElementById('documentList');
    const pdfViewer = document.getElementById('pdfViewer');
    const pdfCanvas = document.getElementById('pdfCanvas');
    const closePdfBtn = document.getElementById('closePdf');
    const prevPageBtn = document.getElementById('prevPage');
    const nextPageBtn = document.getElementById('nextPage');
    const pageInfoSpan = document.getElementById('pageInfo');
    const presentationViewer = document.getElementById('presentationViewer');
    const slideImage = document.getElementById('slideImage');
    const closePresentationBtn = document.getElementById('closePresentation');
    const prevSlideBtn = document.getElementById('prevSlide');
    const nextSlideBtn = document.getElementById('nextSlide');
    const slideInfoSpan = document.getElementById('slideInfo');
    
    // Variables
    let socket = null;
    let activeTouches = new Map();
    let currentTool = 'pen';
    let currentColor = '#000000';
    let currentBrushSize = 5;
    let isDrawing = false;
    let lastPoint = null;
    let drawingObject = null;
    let undoStack = [];
    let redoStack = [];
    let pdfDocument = null;
    let currentPage = 1;
    let presentationSlides = [];
    let currentSlide = 0;
    let loadedDocuments = [];
    
    // Initialize
    init();
    
    function init() {
        setupCanvas();
        setupSocketIO();
        setupEventListeners();
        loadStoredDocuments();
        resizeCanvas();
    }
    
    function setupCanvas() {
        // Set canvas background
        canvas.setBackgroundColor('#ffffff', canvas.renderAll.bind(canvas));
        
        // Set initial dimensions
        resizeCanvas();
        
        // Setup drawing modes
        setupDrawingModes();
        
        // Save initial state
        saveState();
    }
    
    function setupSocketIO() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        socket = io.connect(`${protocol}//${host}`);
        
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('touch_update', function(data) {
            updateTouchIndicators(data.touches);
        });
        
        socket.on('draw_update', function(data) {
            handleRemoteDrawAction(data);
        });
        
        socket.on('canvas_cleared', function() {
            clearCanvas();
        });
        
        socket.on('document_loaded', function(data) {
            loadDocumentFromData(data);
        });
    }
    
    function setupEventListeners() {
        // Window resize
        window.addEventListener('resize', resizeCanvas);
        
        // Tool selection
        toolButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                toolButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                currentTool = this.dataset.tool;
                updateCursor();
            });
        });
        
        // Color selection
        colorOptions.forEach(option => {
            option.addEventListener('click', function() {
                colorOptions.forEach(o => o.classList.remove('active'));
                this.classList.add('active');
                currentColor = this.dataset.color;
                customColorInput.value = currentColor;
            });
        });
        
        customColorInput.addEventListener('change', function() {
            currentColor = this.value;
            colorOptions.forEach(o => o.classList.remove('active'));
        });
        
        // Brush size
        brushSizeSlider.addEventListener('input', function() {
            currentBrushSize = parseInt(this.value);
            brushSizeValueSpan.textContent = currentBrushSize + 'px';
            updateCursor();
        });
        
        // Canvas controls
        clearCanvasBtn.addEventListener('click', clearCanvas);
        undoBtn.addEventListener('click', undo);
        redoBtn.addEventListener('click', redo);
        saveCanvasBtn.addEventListener('click', saveCanvasAsImage);
        fullscreenBtn.addEventListener('click', toggleFullscreen);
        
        // Document upload
        uploadPdfBtn.addEventListener('click', () => openFileDialog('pdf'));
        uploadPresentationBtn.addEventListener('click', () => openFileDialog('presentation'));
        fileInput.addEventListener('change', handleFileUpload);
        
        // PDF viewer controls
        closePdfBtn.addEventListener('click', closePdfViewer);
        prevPageBtn.addEventListener('click', prevPage);
        nextPageBtn.addEventListener('click', nextPage);
        
        // Presentation viewer controls
        closePresentationBtn.addEventListener('click', closePresentationViewer);
        prevSlideBtn.addEventListener('click', prevSlide);
        nextSlideBtn.addEventListener('click', nextSlide);
        
        // Touch simulation for testing
        setupTouchSimulation();
    }
    
    function setupDrawingModes() {
        canvas.isDrawingMode = false;
        
        // Free drawing
        canvas.on('mouse:down', function(options) {
            if (currentTool === 'pen' || currentTool === 'eraser') {
                isDrawing = true;
                lastPoint = options.pointer;
                
                if (currentTool === 'pen') {
                    drawingObject = new fabric.Path(`M ${lastPoint.x} ${lastPoint.y}`, {
                        stroke: currentColor,
                        strokeWidth: currentBrushSize,
                        fill: null,
                        strokeLineCap: 'round',
                        strokeLineJoin: 'round'
                    });
                    canvas.add(drawingObject);
                } else if (currentTool === 'eraser') {
                    // Eraser implementation
                    eraseAtPoint(lastPoint);
                }
            } else if (currentTool === 'text') {
                addTextAtPoint(options.pointer);
            }
        });
        
        canvas.on('mouse:move', function(options) {
            if (!isDrawing || !lastPoint) return;
            
            const currentPoint = options.pointer;
            
            if (currentTool === 'pen') {
                drawingObject.path.push(['L', currentPoint.x, currentPoint.y]);
                drawingObject.set({ dirty: true });
                canvas.renderAll();
            } else if (currentTool === 'eraser') {
                eraseAtPoint(currentPoint);
            }
            
            lastPoint = currentPoint;
        });
        
        canvas.on('mouse:up', function() {
            if (isDrawing) {
                isDrawing = false;
                lastPoint = null;
                
                if (drawingObject) {
                    saveState();
                    drawingObject = null;
                }
            }
        });
    }
    
    function updateCursor() {
        const cursorSize = Math.max(currentBrushSize, 10);
        
        if (currentTool === 'pen') {
            canvas.defaultCursor = `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="${cursorSize}" height="${cursorSize}" viewBox="0 0 ${cursorSize} ${cursorSize}"><circle cx="${cursorSize/2}" cy="${cursorSize/2}" r="${cursorSize/2}" fill="${currentColor}" opacity="0.5"/></svg>') ${cursorSize/2} ${cursorSize/2}, auto`;
        } else if (currentTool === 'eraser') {
            canvas.defaultCursor = `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="${cursorSize}" height="${cursorSize}" viewBox="0 0 ${cursorSize} ${cursorSize}"><circle cx="${cursorSize/2}" cy="${cursorSize/2}" r="${cursorSize/2}" fill="white" stroke="black" stroke-width="1" opacity="0.7"/></svg>') ${cursorSize/2} ${cursorSize/2}, auto`;
        } else {
            canvas.defaultCursor = 'default';
        }
    }
    
    function eraseAtPoint(point) {
        const objects = canvas.getObjects();
        const eraserRadius = currentBrushSize;
        
        objects.forEach(obj => {
            if (obj.type === 'path' || obj.type === 'line' || obj.type === 'rect' || obj.type === 'circle' || obj.type === 'text') {
                // Simple distance check for demo
                // In production, use more sophisticated hit testing
                if (obj.intersectsWithRect(new fabric.Rect({
                    left: point.x - eraserRadius,
                    top: point.y - eraserRadius,
                    width: eraserRadius * 2,
                    height: eraserRadius * 2
                }))) {
                    canvas.remove(obj);
                }
            }
        });
        
        canvas.renderAll();
    }
    
    function addTextAtPoint(point) {
        const text = new fabric.IText('Type here', {
            left: point.x,
            top: point.y,
            fontFamily: 'Arial',
            fontSize: 24,
            fill: currentColor
        });
        
        canvas.add(text);
        canvas.setActiveObject(text);
        text.enterEditing();
        saveState();
    }
    
    function updateTouchIndicators(touches) {
        // Clear existing indicators
        touchIndicators.innerHTML = '';
        activeTouches.clear();
        
        // Update touch count
        activeTouchesSpan.textContent = touches.length;
        
        // Create new indicators
        touches.forEach((touch, index) => {
            const indicator = document.createElement('div');
            indicator.className = 'touch-indicator';
            indicator.style.left = `${touch.x * 100}%`;
            indicator.style.top = `${touch.y * 100}%`;
            indicator.textContent = index + 1;
            indicator.style.lineHeight = '40px';
            indicator.style.textAlign = 'center';
            indicator.style.color = 'white';
            indicator.style.fontWeight = 'bold';
            
            touchIndicators.appendChild(indicator);
            activeTouches.set(index, touch);
        });
        
        // Update touch positions display
        updateTouchPositionsDisplay(touches);
        
        // Handle touch gestures for tools
        if (touches.length === 1 && currentTool !== 'pen') {
            // Single touch for drawing with current tool
            const touch = touches[0];
            handleTouchDrawing(touch);
        } else if (touches.length === 2) {
            // Two-finger gesture for pan/zoom
            handleTwoFingerGesture(touches[0], touches[1]);
        }
    }
    
    function handleTouchDrawing(touch) {
        const canvasRect = canvas.getElement().getBoundingClientRect();
        const point = {
            x: touch.x * canvasRect.width,
            y: touch.y * canvasRect.height
        };
        
        if (currentTool === 'line') {
            drawLine(point);
        } else if (currentTool === 'rectangle') {
            drawRectangle(point);
        } else if (currentTool === 'circle') {
            drawCircle(point);
        }
    }
    
    function drawLine(point) {
        if (!isDrawing) {
            isDrawing = true;
            lastPoint = point;
            drawingObject = new fabric.Line([point.x, point.y, point.x, point.y], {
                stroke: currentColor,
                strokeWidth: currentBrushSize
            });
            canvas.add(drawingObject);
        } else {
            drawingObject.set({
                x2: point.x,
                y2: point.y
            });
            canvas.renderAll();
        }
    }
    
    function drawRectangle(point) {
        if (!isDrawing) {
            isDrawing = true;
            lastPoint = point;
            drawingObject = new fabric.Rect({
                left: point.x,
                top: point.y,
                width: 0,
                height: 0,
                fill: 'transparent',
                stroke: currentColor,
                strokeWidth: currentBrushSize
            });
            canvas.add(drawingObject);
        } else {
            const width = point.x - lastPoint.x;
            const height = point.y - lastPoint.y;
            
            drawingObject.set({
                width: Math.abs(width),
                height: Math.abs(height),
                left: width < 0 ? point.x : lastPoint.x,
                top: height < 0 ? point.y : lastPoint.y
            });
            canvas.renderAll();
        }
    }
    
    function drawCircle(point) {
        if (!isDrawing) {
            isDrawing = true;
            lastPoint = point;
            drawingObject = new fabric.Circle({
                left: point.x,
                top: point.y,
                radius: 0,
                fill: 'transparent',
                stroke: currentColor,
                strokeWidth: currentBrushSize
            });
            canvas.add(drawingObject);
        } else {
            const radius = Math.sqrt(
                Math.pow(point.x - lastPoint.x, 2) + 
                Math.pow(point.y - lastPoint.y, 2)
            ) / 2;
            
            drawingObject.set({
                radius: radius,
                left: lastPoint.x - radius,
                top: lastPoint.y - radius
            });
            canvas.renderAll();
        }
    }
    
    function handleTwoFingerGesture(touch1, touch2) {
        // For demo purposes, we'll implement pan/zoom
        // In production, implement proper gesture recognition
        console.log('Two finger gesture detected');
    }
    
    function updateTouchPositionsDisplay(touches) {
        if (touches.length === 0) {
            touchPositionsDiv.innerHTML = 'No touches detected';
            return;
        }
        
        let html = '<ul>';
        touches.forEach((touch, index) => {
            html += `<li>Touch ${index + 1}: X=${touch.x.toFixed(3)}, Y=${touch.y.toFixed(3)}</li>`;
        });
        html += '</ul>';
        touchPositionsDiv.innerHTML = html;
    }
    
    function clearCanvas() {
        if (confirm('Clear the entire canvas?')) {
            canvas.clear();
            canvas.setBackgroundColor('#ffffff', canvas.renderAll.bind(canvas));
            saveState();
            
            // Broadcast to other clients
            if (socket) {
                socket.emit('clear_canvas');
            }
        }
    }
    
    function saveState() {
        const json = JSON.stringify(canvas.toJSON());
        undoStack.push(json);
        redoStack = [];
    }
    
    function undo() {
        if (undoStack.length > 1) {
            redoStack.push(undoStack.pop());
            const state = undoStack[undoStack.length - 1];
            canvas.loadFromJSON(state, canvas.renderAll.bind(canvas));
        }
    }
    
    function redo() {
        if (redoStack.length > 0) {
            const state = redoStack.pop();
            undoStack.push(state);
            canvas.loadFromJSON(state, canvas.renderAll.bind(canvas));
        }
    }
    
    function saveCanvasAsImage() {
        const link = document.createElement('a');
        link.download = `whiteboard-${Date.now()}.png`;
        link.href = canvas.toDataURL({
            format: 'png',
            quality: 1
        });
        link.click();
    }
    
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    }
    
    function openFileDialog(type) {
        fileInput.setAttribute('data-type', type);
        
        if (type === 'pdf') {
            fileInput.accept = '.pdf';
        } else {
            fileInput.accept = '.ppt,.pptx,.png,.jpg,.jpeg';
        }
        
        fileInput.click();
    }
    
    async function handleFileUpload(event) {
        const file = event.target.files[0];
        const type = fileInput.getAttribute('data-type');
        
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Add to document list
                addDocumentToList(result);
                
                // Load the document
                loadDocument(result);
                
                // Broadcast to other clients
                if (socket) {
                    socket.emit('load_document', result);
                }
            } else {
                alert('Error uploading file: ' + result.error);
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Error uploading file');
        }
        
        // Reset file input
        fileInput.value = '';
    }
    
    function addDocumentToList(document) {
        loadedDocuments.push(document);
        
        const docElement = document.createElement('div');
        docElement.className = 'document-item';
        docElement.innerHTML = `
            <strong>${document.filename}</strong>
            <button class="btn-load-doc" data-url="${document.url}" data-type="${document.type}">
                <i class="fas fa-eye"></i> Load
            </button>
        `;
        
        documentList.appendChild(docElement);
        
        // Add event listener to load button
        docElement.querySelector('.btn-load-doc').addEventListener('click', function() {
            loadDocument({
                url: this.dataset.url,
                type: this.dataset.type,
                filename: this.parentElement.querySelector('strong').textContent
            });
        });
    }
    
    function loadDocument(document) {
        if (document.type === 'pdf') {
            openPdfViewer(document.url);
        } else {
            openPresentationViewer(document.url);
        }
    }
    
    function loadDocumentFromData(document) {
        addDocumentToList(document);
        loadDocument(document);
    }
    
    function loadStoredDocuments() {
        // In production, load from server/localStorage
        // For demo, we'll start with an empty list
        documentList.innerHTML = '<p>No documents loaded</p>';
    }
    
    function openPdfViewer(url) {
        // For demo, we'll show a placeholder
        // In production, integrate with PDF.js or similar library
        pdfViewer.style.display = 'block';
        pageInfoSpan.textContent = 'Page 1 of 1';
        
        // Load PDF using PDF.js would go here
        console.log('Loading PDF from:', url);
    }
    
    function closePdfViewer() {
        pdfViewer.style.display = 'none';
    }
    
    function prevPage() {
        if (pdfDocument && currentPage > 1) {
            currentPage--;
            renderPdfPage();
        }
    }
    
    function nextPage() {
        if (pdfDocument && currentPage < pdfDocument.numPages) {
            currentPage++;
            renderPdfPage();
        }
    }
    
    function renderPdfPage() {
        // PDF rendering implementation would go here
        pageInfoSpan.textContent = `Page ${currentPage} of ${pdfDocument ? pdfDocument.numPages : 1}`;
    }
    
    function openPresentationViewer(url) {
        presentationViewer.style.display = 'block';
        slideImage.src = url;
        slideInfoSpan.textContent = 'Slide 1 of 1';
        currentSlide = 0;
        presentationSlides = [url]; // In production, extract slides from PPT
    }
    
    function closePresentationViewer() {
        presentationViewer.style.display = 'none';
    }
    
    function prevSlide() {
        if (currentSlide > 0) {
            currentSlide--;
            slideImage.src = presentationSlides[currentSlide];
            slideInfoSpan.textContent = `Slide ${currentSlide + 1} of ${presentationSlides.length}`;
        }
    }
    
    function nextSlide() {
        if (currentSlide < presentationSlides.length - 1) {
            currentSlide++;
            slideImage.src = presentationSlides[currentSlide];
            slideInfoSpan.textContent = `Slide ${currentSlide + 1} of ${presentationSlides.length}`;
        }
    }
    
    function handleRemoteDrawAction(data) {
        // Handle drawing actions from other clients
        // Implementation depends on your data structure
        console.log('Remote draw action:', data);
    }
    
    function resizeCanvas() {
        const container = canvas.getElement().parentElement;
        canvas.setWidth(container.offsetWidth);
        canvas.setHeight(container.offsetHeight);
        canvas.renderAll();
    }
    
    function setupTouchSimulation() {
        // For testing without camera
        document.addEventListener('keydown', function(e) {
            if (e.key === 't') {
                // Simulate a touch
                const simulatedTouch = {
                    x: Math.random(),
                    y: Math.random()
                };
                updateTouchIndicators([simulatedTouch]);
            }
        });
    }
});