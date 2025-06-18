"""
Utilidades para el clasificador de frutas - CORREGIDO PARA 150x150
Compatible con main.py y el modelo best_model.keras
"""
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, List, Optional
from datetime import datetime

# ✅ CONFIGURACIÓN CORREGIDA - Compatible con main.py
DEFAULT_IMAGE_SIZE = (150, 150)  # ✅ CORREGIDO: Era (224, 224)
INPUT_SHAPE = (150, 150, 3)  # ✅ Shape esperado por el modelo
SUPPORTED_FORMATS = ['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF', 'WEBP']

# ✅ Lista de frutas actualizada - Compatible con main.py
FRUIT_CLASSES = [
    "Pera", "Ciruela", "Manzana", "Naranja"
]

def validate_image(image_data: bytes) -> bool:
    """
    Valida que los datos correspondan a una imagen válida
    
    Args:
        image_data: Datos binarios de la imagen
        
    Returns:
        bool: True si es una imagen válida
    """
    try:
        if len(image_data) == 0:
            print("❌ Imagen vacía")
            return False
        
        image = Image.open(io.BytesIO(image_data))
        
        # Verificar formato soportado
        if image.format not in SUPPORTED_FORMATS:
            print(f"❌ Formato no soportado: {image.format}")
            return False
        
        # Verificar dimensiones mínimas
        if image.size[0] < 32 or image.size[1] < 32:
            print(f"❌ Imagen demasiado pequeña: {image.size}")
            return False
        
        print(f"✅ Imagen válida: {image.size}, formato: {image.format}")
        return True
        
    except Exception as e:
        print(f"❌ Error validando imagen: {e}")
        return False

def preprocess_image_for_model(image: Image.Image, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """
    ✅ FUNCIÓN PRINCIPAL - Preprocesa imagen exactamente como main.py
    
    Args:
        image: Imagen PIL
        target_size: Tupla (ancho, alto) del tamaño objetivo (150, 150)
        
    Returns:
        np.ndarray: Array listo para el modelo (1, 150, 150, 3)
    """
    try:
        print(f"🔄 === PREPROCESANDO IMAGEN (UTILS) ===")
        print(f"📷 Imagen original: {image.size} píxeles, modo: {image.mode}")
        
        # 1. Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"🔄 Convertida a RGB")
        
        # 2. ✅ CRÍTICO: Redimensionar a 150x150 (mismo método que main.py)
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        print(f"📐 Redimensionada a: {image_resized.size} (CORRECTO: 150x150)")
        
        # 3. Convertir a array numpy
        img_array = np.array(image_resized, dtype=np.float32)
        print(f"🔢 Array shape: {img_array.shape}")
        
        # 4. Normalizar píxeles (0-255 → 0-1)
        img_array = img_array / 255.0
        print(f"📊 Normalizado: min={img_array.min():.3f}, max={img_array.max():.3f}")
        
        # 5. Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        print(f"🎯 Shape final para modelo: {img_array.shape} (esperado: (1, 150, 150, 3))")
        
        # 6. Verificación final
        expected_shape = (1, 150, 150, 3)
        if img_array.shape == expected_shape:
            print("✅ Shape CORRECTO para el modelo")
            return img_array
        else:
            raise ValueError(f"Shape incorrecto: esperado {expected_shape}, obtenido {img_array.shape}")
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento (utils): {str(e)}")
        raise e

def process_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    ✅ Procesa bytes de imagen directamente - Compatible con main.py
    
    Args:
        image_bytes: Bytes de la imagen
        
    Returns:
        np.ndarray: Array listo para el modelo
    """
    try:
        # Validar imagen
        if not validate_image(image_bytes):
            raise ValueError("Imagen no válida")
        
        # Abrir imagen
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocesar para el modelo
        return preprocess_image_for_model(image)
        
    except Exception as e:
        print(f"❌ Error procesando bytes de imagen: {e}")
        raise e

def resize_image_maintain_aspect(image: Image.Image, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """
    ✅ Redimensiona imagen manteniendo proporción con padding blanco
    
    Args:
        image: Imagen PIL
        target_size: Tupla (ancho, alto) del tamaño objetivo
        
    Returns:
        Image.Image: Imagen redimensionada con padding si es necesario
    """
    try:
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calcular el ratio de escalado manteniendo proporción
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_ratio = min(width_ratio, height_ratio)
        
        # Calcular el nuevo tamaño
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        
        # Redimensionar la imagen
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crear imagen final con fondo blanco
        final_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Centrar la imagen redimensionada
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_image.paste(resized_image, (paste_x, paste_y))
        
        return final_image
        
    except Exception as e:
        print(f"❌ Error redimensionando imagen: {e}")
        raise e

def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decodifica una imagen en base64
    
    Args:
        base64_string: String en base64
        
    Returns:
        Image.Image: Imagen PIL
    """
    try:
        # Remover el prefijo data:image si existe
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decodificar
        image_data = base64.b64decode(base64_string)
        
        # Validar antes de abrir
        if not validate_image(image_data):
            raise ValueError("Imagen base64 no válida")
        
        image = Image.open(io.BytesIO(image_data))
        return image
        
    except Exception as e:
        raise ValueError(f"Error al decodificar imagen base64: {e}")

def get_image_info(image: Image.Image) -> dict:
    """
    Obtiene información detallada sobre una imagen
    
    Args:
        image: Imagen PIL
        
    Returns:
        dict: Información completa de la imagen
    """
    try:
        info = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "has_transparency": image.mode in ('RGBA', 'LA', 'P'),
        }
        
        # Calcular tamaño en bytes aproximado
        if hasattr(image, 'info'):
            info["info"] = image.info
        
        return info
        
    except Exception as e:
        print(f"❌ Error obteniendo info de imagen: {e}")
        return {"error": str(e)}

def create_prediction_response(fruit_name: str, confidence: float, 
                             processing_time: Optional[float] = None,
                             image_info: Optional[dict] = None,
                             model_info: Optional[dict] = None) -> dict:
    """
    ✅ Crea respuesta estructurada compatible con main.py
    
    Args:
        fruit_name: Nombre de la fruta predicha
        confidence: Confianza de la predicción (0-100)
        processing_time: Tiempo de procesamiento en ms
        image_info: Información de la imagen
        model_info: Información del modelo
        
    Returns:
        dict: Respuesta estructurada compatible con frontend
    """
    response = {
        "fruit": fruit_name,
        "confidence": round(confidence, 2),
        "status": "success",
        "model_loaded": True,
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "input_shape": list(INPUT_SHAPE),
        "timestamp": datetime.now().isoformat()
    }
    
    if processing_time is not None:
        response["processing_time"] = f"{processing_time:.2f}ms"
    
    if image_info:
        response["image_info"] = image_info
    
    if model_info:
        response["model_info"] = model_info
    
    return response

def get_top_predictions(predictions: np.ndarray, class_names: List[str] = FRUIT_CLASSES, top_k: int = 3) -> List[dict]:
    """
    ✅ Obtiene las mejores predicciones - Compatible con el modelo actual
    
    Args:
        predictions: Array de predicciones del modelo (shape: (1, num_classes))
        class_names: Lista de nombres de clases
        top_k: Número de mejores predicciones a retornar
        
    Returns:
        List[dict]: Lista de mejores predicciones ordenadas por confianza
    """
    try:
        # Verificar que las predicciones tengan el formato correcto
        if len(predictions.shape) != 2 or predictions.shape[0] != 1:
            raise ValueError(f"Formato de predicciones incorrecto: {predictions.shape}")
        
        # Obtener las probabilidades de la primera (y única) muestra
        probs = predictions[0]
        
        # Obtener los índices ordenados por probabilidad (descendente)
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(class_names):
                confidence = float(probs[idx]) * 100
                results.append({
                    "rank": i + 1,
                    "fruit": class_names[idx],
                    "confidence": round(confidence, 2),
                    "index": int(idx)
                })
        
        return results
        
    except Exception as e:
        print(f"❌ Error obteniendo top predicciones: {e}")
        return []

def log_prediction(image_info: dict, prediction: dict, success: bool = True, processing_time: Optional[float] = None):
    """
    ✅ Registra información sobre una predicción - Mejorado para debugging
    
    Args:
        image_info: Información de la imagen
        prediction: Resultado de la predicción
        success: Si la predicción fue exitosa
        processing_time: Tiempo de procesamiento en ms
    """
    status = "✅ ÉXITO" if success else "❌ ERROR"
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"📊 === REGISTRO DE PREDICCIÓN [{timestamp}] ===")
    print(f"   {status}")
    print(f"   📸 Imagen: {image_info.get('size', 'unknown')} píxeles")
    print(f"   📋 Formato: {image_info.get('format', 'unknown')}")
    print(f"   🍎 Fruta predicha: {prediction.get('fruit', 'unknown')}")
    print(f"   📈 Confianza: {prediction.get('confidence', 0)}%")
    
    if processing_time:
        print(f"   ⏱️  Tiempo: {processing_time:.2f}ms")
    
    print("=" * 50)

def validate_model_compatibility(model_input_shape: tuple) -> bool:
    """
    ✅ Valida que el modelo sea compatible con las utilidades
    
    Args:
        model_input_shape: Shape de entrada del modelo cargado
        
    Returns:
        bool: True si es compatible
    """
    expected_shape = (None, 150, 150, 3)
    
    try:
        if model_input_shape == expected_shape:
            print("✅ Modelo compatible con utils.py")
            return True
        else:
            print(f"⚠️  Advertencia: Shape esperado {expected_shape}, encontrado {model_input_shape}")
            # Permitir cierta flexibilidad
            if (model_input_shape[1:] == (150, 150, 3)):
                print("✅ Dimensiones compatibles (ignorando batch size)")
                return True
            else:
                print("❌ Modelo incompatible")
                return False
                
    except Exception as e:
        print(f"❌ Error validando compatibilidad: {e}")
        return False

def get_system_info() -> dict:
    """
    ✅ Información del sistema de utilidades
    
    Returns:
        dict: Información del sistema
    """
    return {
        "utils_version": "2.0.0",
        "image_size": DEFAULT_IMAGE_SIZE,
        "input_shape": INPUT_SHAPE,
        "supported_formats": SUPPORTED_FORMATS,
        "fruit_classes": FRUIT_CLASSES,
        "num_classes": len(FRUIT_CLASSES),
        "compatible_with": "main.py v1.0.0 + best_model.keras (150x150)",
        "last_updated": "2025-06-18"
    }

# ✅ FUNCIONES DE CONVENIENCIA PARA INTEGRACIÓN RÁPIDA

def quick_preprocess(image_bytes: bytes) -> np.ndarray:
    """
    ✅ Función rápida para preprocesar imagen - Un solo paso
    
    Args:
        image_bytes: Bytes de la imagen
        
    Returns:
        np.ndarray: Array listo para predicción
    """
    return process_image_bytes(image_bytes)

def quick_prediction_response(fruit: str, confidence: float, time_ms: float = None) -> dict:
    """
    ✅ Respuesta rápida para predicciones - Compatible con frontend
    
    Args:
        fruit: Nombre de la fruta
        confidence: Confianza (0-100)
        time_ms: Tiempo en milisegundos
        
    Returns:
        dict: Respuesta lista para JSON
    """
    return create_prediction_response(fruit, confidence, time_ms)

# ✅ VERIFICACIÓN AL IMPORTAR
if __name__ == "__main__":
    print("🔧 === UTILS.PY PARA CLASIFICADOR DE FRUTAS ===")
    info = get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print("✅ Utils cargado correctamente")
    print("=" * 50)