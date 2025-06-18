// Configuración de la API
const API_BASE_URL = 'http://127.0.0.1:8000';

// Variables globales
let currentStream = null;
let currentMode = 'upload';
let debugMode = true; // Activar logging detallado

// Referencias a elementos del DOM
const uploadBtn = document.getElementById('uploadBtn');
const cameraBtn = document.getElementById('cameraBtn');
const fileInput = document.getElementById('fileInput');
const methodBtn = document.getElementById('methodBtn');
const resultArea = document.getElementById('resultArea');
const previewImage = document.getElementById('previewImage');
const loadingOverlay = document.getElementById('loadingOverlay');
const predictionTitle = document.getElementById('predictionTitle');
const confidenceText = document.getElementById('confidenceText');
const cameraContainer = document.getElementById('cameraContainer');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Funciones de logging
function debugLog(message, data = null) {
    if (debugMode) {
        console.log(`🔍 DEBUG: ${message}`, data || '');
    }
}

function errorLog(message, error = null) {
    console.error(`❌ ERROR: ${message}`, error || '');
}

function successLog(message, data = null) {
    console.log(`✅ SUCCESS: ${message}`, data || '');
}

// Inicializar la aplicación
document.addEventListener('DOMContentLoaded', function() {
    debugLog('DOM cargado, inicializando aplicación...');
    
    // Crear panel de debug
    createDebugPanel();
    
    initializeEventListeners();
    checkAPIConnection();
    runSystemDiagnostics();
});

function createDebugPanel() {
    // Crear panel de debug solo si estamos en localhost
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debugPanel';
        debugPanel.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            width: 300px;
            max-height: 400px;
            overflow-y: auto;
            background: rgba(0,0,0,0.9);
            color: #00ff00;
            font-family: monospace;
            font-size: 12px;
            padding: 10px;
            border-radius: 5px;
            z-index: 9999;
            display: none;
        `;
        document.body.appendChild(debugPanel);
        
        // Botón para mostrar/ocultar debug
        const debugToggle = document.createElement('button');
        debugToggle.textContent = '🐛 Debug';
        debugToggle.style.cssText = `
            position: fixed;
            top: 10px;
            right: 320px;
            z-index: 10000;
            padding: 5px 10px;
            background: #333;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        `;
        debugToggle.onclick = () => {
            const panel = document.getElementById('debugPanel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        };
        document.body.appendChild(debugToggle);
    }
}

function addToDebugPanel(message) {
    const panel = document.getElementById('debugPanel');
    if (panel) {
        const timestamp = new Date().toLocaleTimeString();
        panel.innerHTML += `<div>[${timestamp}] ${message}</div>`;
        panel.scrollTop = panel.scrollHeight;
    }
}

function initializeEventListeners() {
    debugLog('Inicializando event listeners...');
    
    // Botón subir imagen
    uploadBtn.addEventListener('click', () => {
        debugLog('Click en botón upload');
        addToDebugPanel('🖱️ Click en botón upload');
        if (currentMode === 'upload') {
            fileInput.click();
        } else {
            switchToUploadMode();
        }
    });

    // Botón cámara
    cameraBtn.addEventListener('click', () => {
        debugLog('Click en botón cámara');
        addToDebugPanel('🖱️ Click en botón cámara');
        if (currentMode === 'camera') {
            startCamera();
        } else {
            switchToCameraMode();
        }
    });

    // Input de archivo
    fileInput.addEventListener('change', (event) => {
        debugLog('Archivo seleccionado', event.target.files[0]);
        addToDebugPanel('📁 Archivo seleccionado: ' + (event.target.files[0]?.name || 'ninguno'));
        handleFileSelect(event);
    });

    // Botón cambiar método
    methodBtn.addEventListener('click', toggleMethod);

    // Controles de cámara
    captureBtn.addEventListener('click', capturePhoto);
    closeCameraBtn.addEventListener('click', stopCamera);

    // Drag and drop
    document.addEventListener('dragover', handleDragOver);
    document.addEventListener('drop', handleDrop);
    
    successLog('Event listeners inicializados');
}

async function checkAPIConnection() {
    debugLog('Verificando conexión con API...');
    addToDebugPanel('🔗 Verificando conexión API...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            }
        });
        
        debugLog('Respuesta de health check', {
            status: response.status,
            statusText: response.statusText,
            headers: Object.fromEntries(response.headers.entries())
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        successLog('Conectado al backend', data);
        addToDebugPanel('✅ Backend conectado: ' + data.mode);
        hideError();
        
        return data;
    } catch (error) {
        errorLog('Error de conexión', error);
        addToDebugPanel('❌ Error conexión: ' + error.message);
        showError('No se pudo conectar con el servidor. Asegúrate de que el backend esté ejecutándose en http://127.0.0.1:8000');
        return null;
    }
}

function runSystemDiagnostics() {
    debugLog('Ejecutando diagnósticos del sistema...');
    
    const diagnostics = {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        cookieEnabled: navigator.cookieEnabled,
        onLine: navigator.onLine,
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        fileAPI: !!(window.File && window.FileReader && window.FileList && window.Blob),
        canvas: !!document.createElement('canvas').getContext,
        fetch: typeof fetch !== 'undefined',
        formData: typeof FormData !== 'undefined'
    };
    
    debugLog('Diagnósticos del sistema', diagnostics);
    
    // Mostrar en panel de debug
    Object.entries(diagnostics).forEach(([key, value]) => {
        const status = value ? '✅' : '❌';
        addToDebugPanel(`${status} ${key}: ${value}`);
    });
}

function switchToUploadMode() {
    currentMode = 'upload';
    uploadBtn.classList.add('active');
    cameraBtn.classList.remove('active');
    hideCameraContainer();
    debugLog('Cambiado a modo upload');
}

function switchToCameraMode() {
    currentMode = 'camera';
    cameraBtn.classList.add('active');
    uploadBtn.classList.remove('active');
    debugLog('Cambiado a modo cámara');
}

function toggleMethod() {
    if (currentMode === 'upload') {
        switchToCameraMode();
    } else {
        switchToUploadMode();
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        debugLog('Archivo seleccionado para procesar', {
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified
        });
        processSelectedFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
}

function handleDrop(event) {
    event.preventDefault();
    const files = event.dataTransfer.files;
    debugLog('Archivos soltados', files);
    if (files.length > 0) {
        processSelectedFile(files[0]);
    }
}

function processSelectedFile(file) {
    debugLog('Procesando archivo seleccionado', {
        name: file.name,
        size: formatBytes(file.size),
        type: file.type
    });
    
    addToDebugPanel(`📁 Procesando: ${file.name} (${formatBytes(file.size)})`);
    
    // Validaciones más detalladas
    if (!file.type.startsWith('image/')) {
        const error = `Tipo de archivo inválido: ${file.type}`;
        errorLog(error);
        addToDebugPanel('❌ ' + error);
        showError('❌ Por favor selecciona un archivo de imagen válido (JPG, PNG, etc.)');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        const error = `Archivo demasiado grande: ${formatBytes(file.size)}`;
        errorLog(error);
        addToDebugPanel('❌ ' + error);
        showError('❌ La imagen es demasiado grande. Máximo 10MB permitido');
        return;
    }

    if (file.size < 1024) {
        const error = 'Archivo demasiado pequeño';
        errorLog(error);
        addToDebugPanel('❌ ' + error);
        showError('❌ La imagen es demasiado pequeña');
        return;
    }

    // Mostrar preview y clasificar
    const reader = new FileReader();
    reader.onload = function(e) {
        debugLog('FileReader completado');
        addToDebugPanel('📖 Archivo leído correctamente');
        showImagePreview(e.target.result);
        classifyImage(file);
    };
    reader.onerror = function(e) {
        errorLog('Error en FileReader', e);
        addToDebugPanel('❌ Error leyendo archivo');
        showError('❌ Error al leer el archivo');
    };
    reader.readAsDataURL(file);
}

function showImagePreview(imageUrl) {
    debugLog('Mostrando preview de imagen');
    previewImage.src = imageUrl;
    previewImage.onload = () => {
        debugLog('Preview cargado', {
            naturalWidth: previewImage.naturalWidth,
            naturalHeight: previewImage.naturalHeight
        });
        addToDebugPanel(`🖼️ Preview: ${previewImage.naturalWidth}x${previewImage.naturalHeight}`);
        resultArea.style.display = 'block';
        resultArea.classList.add('fade-in');
    };
    hideError();
}

async function classifyImage(file) {
    debugLog('Iniciando clasificación de imagen', {
        name: file.name,
        size: file.size,
        type: file.type
    });
    
    addToDebugPanel(`🤖 Clasificando: ${file.name}`);
    
    // Mostrar loading
    showLoading();
    resetPrediction();

    try {
        // Crear FormData
        const formData = new FormData();
        formData.append('file', file);
        
        debugLog('FormData creado', formData);
        addToDebugPanel('📤 Enviando al backend...');

        // Realizar petición con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            addToDebugPanel('⏰ Timeout - petición cancelada');
        }, 30000);

        const startTime = Date.now();
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);
        const requestTime = Date.now() - startTime;
        
        debugLog('Respuesta recibida', {
            status: response.status,
            statusText: response.statusText,
            time: requestTime + 'ms',
            headers: Object.fromEntries(response.headers.entries())
        });
        
        addToDebugPanel(`📡 Respuesta: ${response.status} (${requestTime}ms)`);

        if (!response.ok) {
            const errorText = await response.text();
            const error = `Error ${response.status}: ${response.statusText}\n${errorText}`;
            errorLog(error);
            addToDebugPanel(`❌ HTTP Error: ${response.status}`);
            throw new Error(error);
        }

        const result = await response.json();
        
        debugLog('Resultado de clasificación', result);
        addToDebugPanel(`🎯 Resultado: ${result.fruit} (${result.confidence}%)`);
        
        if (result.status === 'success' && result.fruit) {
            successLog('Clasificación exitosa', result);
            displayPrediction(result);
        } else {
            throw new Error('Respuesta inválida del servidor: ' + JSON.stringify(result));
        }

    } catch (error) {
        errorLog('Error en clasificación', error);
        addToDebugPanel(`❌ Error: ${error.message}`);
        
        let errorMsg = 'Error al clasificar la imagen. ';
        
        if (error.name === 'AbortError') {
            errorMsg += 'La petición tardó demasiado tiempo.';
        } else if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
            errorMsg += 'Problemas de conexión con el servidor.';
        } else if (error.message.includes('CORS')) {
            errorMsg += 'Error de CORS. Verifica la configuración del servidor.';
        } else {
            errorMsg += 'Detalles: ' + error.message;
        }
        
        showError(errorMsg);
    } finally {
        hideLoading();
    }
}

function displayPrediction(result) {
    debugLog('Mostrando predicción', result);
    
    // Emojis de frutas
    const fruitEmojis = {
        'manzana': '🍎', 'banana': '🍌', 'naranja': '🍊', 'pera': '🍐',
        'uva': '🍇', 'fresa': '🍓', 'piña': '🍍', 'sandía': '🍉',
        'melón': '🍈', 'durazno': '🍑', 'cereza': '🍒', 'kiwi': '🥝',
        'mango': '🥭', 'aguacate': '🥑', 'limón': '🍋'
    };
    
    const fruitName = result.fruit.toLowerCase();
    const emoji = fruitEmojis[fruitName] || '🍎';
    
    predictionTitle.textContent = `${emoji} ${result.fruit}`;
    
    // Mostrar confianza
    const confidence = Math.round(result.confidence * 100) / 100;
    confidenceText.textContent = `${confidence}%`;
    
    // Color según confianza
    if (confidence >= 80) {
        confidenceText.style.color = '#4CAF50';
        confidenceText.style.fontWeight = 'bold';
    } else if (confidence >= 60) {
        confidenceText.style.color = '#FF9800';
        confidenceText.style.fontWeight = 'bold';
    } else {
        confidenceText.style.color = '#f44336';
        confidenceText.style.fontWeight = 'normal';
    }
    
    addToDebugPanel(`✨ Mostrado: ${result.fruit} (${confidence}%)`);
}

// Funciones auxiliares
function resetPrediction() {
    predictionTitle.textContent = '';
    confidenceText.textContent = '';
}

function showLoading() {
    if (loadingOverlay) {
        loadingOverlay.classList.add('show');
        debugLog('Mostrando loading...');
    }
}

function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.classList.remove('show');
        debugLog('Ocultando loading...');
    }
}

function showError(message) {
    errorLog('Mostrando error al usuario', message);
    addToDebugPanel(`🚨 Error mostrado: ${message}`);
    
    if (errorText && errorMessage) {
        errorText.textContent = message;
        errorMessage.style.display = 'block';
        errorMessage.classList.add('fade-in');
        
        setTimeout(() => {
            hideError();
        }, 15000); // 15 segundos
    }
}

function hideError() {
    if (errorMessage) {
        errorMessage.style.display = 'none';
        errorMessage.classList.remove('fade-in');
    }
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Funciones de cámara (simplificadas para este test)
async function startCamera() {
    debugLog('Iniciando cámara...');
    addToDebugPanel('📷 Iniciando cámara...');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        
        currentStream = stream;
        video.srcObject = stream;
        showCameraContainer();
        addToDebugPanel('✅ Cámara iniciada');
        
    } catch (error) {
        errorLog('Error iniciando cámara', error);
        addToDebugPanel('❌ Error cámara: ' + error.message);
        showError('Error accediendo a la cámara: ' + error.message);
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        debugLog('Cámara detenida');
        addToDebugPanel('🔴 Cámara detenida');
    }
    hideCameraContainer();
}

function capturePhoto() {
    if (!currentStream) return;
    
    debugLog('Capturando foto...');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(function(blob) {
        const file = new File([blob], 'captura.jpg', { type: 'image/jpeg' });
        const imageUrl = canvas.toDataURL('image/jpeg', 0.8);
        
        showImagePreview(imageUrl);
        classifyImage(file);
        stopCamera();
        
        addToDebugPanel('📸 Foto capturada y enviada');
    }, 'image/jpeg', 0.8);
}

function showCameraContainer() {
    cameraContainer.style.display = 'block';
}

function hideCameraContainer() {
    cameraContainer.style.display = 'none';
}

// Cleanup
window.addEventListener('beforeunload', function() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
});