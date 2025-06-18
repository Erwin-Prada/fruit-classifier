from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import time
from datetime import datetime
import os

# Crear la aplicación FastAPI
app = FastAPI(title="Clasificador de Frutas con IA", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción usar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del modelo
MODEL_PATH = "model/best_model.keras"
IMAGE_SIZE = (150, 150)  # ✅ CORREGIDO: Tamaño correcto del modelo
INPUT_SHAPE = (150, 150, 3)  # ✅ Shape esperado por el modelo

# Lista de clases de frutas (ajusta según tu modelo)
FRUIT_CLASSES = [
    "Pera", "Ciruela", "Manzana", "Naranja"
]

# Variables globales
model = None
model_loaded = False

def load_model():
    """Carga el modelo TensorFlow"""
    global model, model_loaded
    
    try:
        print("🤖 === CARGANDO MODELO DE TENSORFLOW ===")
        print(f"📁 Ruta del modelo: {MODEL_PATH}")
        print(f"📐 Tamaño de entrada esperado: {INPUT_SHAPE}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ ERROR: No se encontró el modelo en {MODEL_PATH}")
            model_loaded = False
            return False
        
        # Cargar el modelo
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Verificar la forma de entrada del modelo
        input_shape = model.input_shape
        print(f"✅ Modelo cargado exitosamente")
        print(f"📊 Shape de entrada del modelo: {input_shape}")
        print(f"🎯 Clases disponibles: {len(FRUIT_CLASSES)}")
        
        # Verificar compatibilidad
        expected_shape = (None, 150, 150, 3)
        if input_shape == expected_shape:
            print("✅ Compatibilidad de dimensiones VERIFICADA")
            model_loaded = True
            return True
        else:
            print(f"⚠️  ADVERTENCIA: Shape esperado {expected_shape}, encontrado {input_shape}")
            model_loaded = True  # Continuar de todas formas
            return True
            
    except Exception as e:
        print(f"❌ ERROR cargando modelo: {str(e)}")
        model_loaded = False
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    ✅ FUNCIÓN CORREGIDA: Preprocesa imagen para el modelo (150x150)
    """
    try:
        print(f"🔄 === PREPROCESANDO IMAGEN ===")
        
        # 1. Abrir la imagen
        image = Image.open(io.BytesIO(image_bytes))
        print(f"📷 Imagen original: {image.size} píxeles, modo: {image.mode}")
        
        # 2. Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"🔄 Convertida a RGB")
        
        # 3. ✅ CORRECCIÓN CRÍTICA: Redimensionar a 150x150 (NO 224x224)
        image_resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        print(f"📐 Redimensionada a: {image_resized.size} (CORRECTO: 150x150)")
        
        # 4. Convertir a array numpy
        img_array = np.array(image_resized, dtype=np.float32)
        print(f"🔢 Array shape: {img_array.shape}")
        
        # 5. Normalizar píxeles (0-255 → 0-1)
        img_array = img_array / 255.0
        print(f"📊 Normalizado: min={img_array.min():.3f}, max={img_array.max():.3f}")
        
        # 6. Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        print(f"🎯 Shape final para modelo: {img_array.shape} (esperado: (1, 150, 150, 3))")
        
        # 7. Verificación final
        expected_shape = (1, 150, 150, 3)
        if img_array.shape == expected_shape:
            print("✅ Shape CORRECTO para el modelo")
            return img_array
        else:
            raise ValueError(f"Shape incorrecto: esperado {expected_shape}, obtenido {img_array.shape}")
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {str(e)}")
        raise e

def make_prediction(img_array: np.ndarray) -> tuple:
    """
    Realiza la predicción usando el modelo cargado
    """
    try:
        print("🤖 === REALIZANDO PREDICCIÓN ===")
        
        if not model_loaded or model is None:
            raise ValueError("Modelo no cargado correctamente")
        
        # Realizar predicción
        start_time = time.time()
        predictions = model.predict(img_array, verbose=0)
        prediction_time = (time.time() - start_time) * 1000  # en ms
        
        print(f"⏱️  Tiempo de predicción: {prediction_time:.2f}ms")
        print(f"📊 Shape de predicciones: {predictions.shape}")
        
        # Obtener la clase con mayor probabilidad
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Verificar índice válido
        if predicted_class_idx >= len(FRUIT_CLASSES):
            raise ValueError(f"Índice de clase inválido: {predicted_class_idx}")
        
        predicted_fruit = FRUIT_CLASSES[predicted_class_idx]
        confidence_percentage = confidence * 100
        
        print(f"🎯 Predicción: {predicted_fruit}")
        print(f"📈 Confianza: {confidence_percentage:.2f}%")
        print(f"🔢 Índice de clase: {predicted_class_idx}")
        
        return predicted_fruit, confidence_percentage, prediction_time
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise e

# Inicializar la aplicación
@app.on_event("startup")
async def startup_event():
    """Evento de inicio: cargar el modelo"""
    print("🚀 === INICIANDO CLASIFICADOR DE FRUTAS ===")
    success = load_model()
    if success:
        print("✅ Sistema listo para clasificar frutas")
    else:
        print("❌ Sistema iniciado con errores en el modelo")
    print("=" * 60)

@app.get("/")
async def root():
    return {
        "message": "🍎 Clasificador de Frutas con IA",
        "status": "running",
        "model_loaded": model_loaded,
        "input_size": list(IMAGE_SIZE),
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "image_size": list(IMAGE_SIZE),
        "input_shape": list(INPUT_SHAPE),
        "available_fruits": len(FRUIT_CLASSES),
        "fruits": FRUIT_CLASSES,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        print(f"\n📤 === NUEVA PREDICCIÓN ===")
        print(f"📁 Archivo: {file.filename}")
        print(f"📋 Tipo MIME: {file.content_type}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        
        # Verificar que el modelo esté cargado
        if not model_loaded or model is None:
            print("❌ Modelo no cargado")
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. Reinicia el servidor."
            )
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith("image/"):
            print(f"❌ Tipo de archivo inválido: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no válido: {file.content_type}. Se requiere una imagen."
            )
        
        # Leer los bytes del archivo
        image_bytes = await file.read()
        file_size_mb = len(image_bytes) / (1024 * 1024)
        print(f"📦 Tamaño del archivo: {file_size_mb:.2f} MB ({len(image_bytes)} bytes)")
        
        # Validar tamaño
        if len(image_bytes) == 0:
            print("❌ Archivo vacío")
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB máximo
            print(f"❌ Archivo demasiado grande: {file_size_mb:.2f} MB")
            raise HTTPException(status_code=400, detail="El archivo es demasiado grande (máximo 10MB)")
        
        # ✅ PREPROCESAMIENTO CORREGIDO
        print("🔄 Preprocesando imagen...")
        img_array = preprocess_image(image_bytes)
        
        # ✅ PREDICCIÓN CON MODELO REAL
        print("🤖 Realizando predicción con modelo...")
        predicted_fruit, confidence, prediction_time = make_prediction(img_array)
        
        # Calcular tiempo total
        total_time = round((time.time() - start_time) * 1000, 2)  # en ms
        
        print(f"✅ === PREDICCIÓN COMPLETADA ===")
        print(f"   🍎 Fruta predicha: {predicted_fruit}")
        print(f"   📊 Confianza: {confidence:.2f}%")
        print(f"   ⏱️  Tiempo total: {total_time}ms")
        print(f"   🎯 Modelo: {MODEL_PATH}")
        
        # ✅ RESPUESTA EN FORMATO ESPERADO POR EL FRONTEND
        response = {
            "fruit": predicted_fruit,
            "confidence": round(confidence, 2),
            "status": "success",
            "model_loaded": True,
            "filename": file.filename,
            "image_size": list(IMAGE_SIZE),
            "processing_time": f"{total_time}ms",
            "prediction_time": f"{prediction_time:.2f}ms",
            "file_size_mb": round(file_size_mb, 2),
            "input_shape": list(INPUT_SHAPE),
            "timestamp": datetime.now().isoformat()
        }
        
        print("📤 Enviando respuesta al frontend...")
        return JSONResponse(content=response)
        
    except HTTPException as he:
        print(f"❌ HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        error_msg = f"Error en la predicción: {str(e)}"
        print(f"❌ Error inesperado: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/test")
async def test_endpoint():
    """Endpoint para probar que el servidor está funcionando"""
    return {
        "message": "✅ Clasificador de Frutas funcionando correctamente",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "fruits": FRUIT_CLASSES,
        "image_size": list(IMAGE_SIZE),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """Información detallada del modelo"""
    if not model_loaded or model is None:
        return {"error": "Modelo no cargado"}
    
    try:
        return {
            "model_loaded": model_loaded,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "expected_size": list(IMAGE_SIZE),
            "classes": FRUIT_CLASSES,
            "num_classes": len(FRUIT_CLASSES),
            "model_path": MODEL_PATH
        }
    except Exception as e:
        return {"error": f"Error obteniendo info del modelo: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 === INICIANDO SERVIDOR DE CLASIFICACIÓN ===")
    print("🌐 URL: http://127.0.0.1:8000")
    print("📖 Docs: http://127.0.0.1:8000/docs")
    print("🧪 Test: http://127.0.0.1:8000/test")
    print("🤖 Model Info: http://127.0.0.1:8000/model-info")
    print(f"📐 Tamaño de imagen: {IMAGE_SIZE}")
    print(f"🍎 Frutas soportadas: {len(FRUIT_CLASSES)}")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )