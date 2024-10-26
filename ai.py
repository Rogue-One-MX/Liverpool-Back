"""AI modules for similarity search."""

import os
import io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()

# Cargar el modelo CLIP y el procesador
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Crear el cliente de Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("PINECONE_API_KEY no está definido.")
pc = Pinecone(api_key=pinecone_api_key)
index_name = os.environ.get("INDEX_NAME", "quickstart")
if index_name is None:
    raise ValueError("INDEX_NAME no está definido.")
index = pc.Index(index_name)


def preprocesar_imagen(img_data, size=(224, 224)):
    """Redimensiona la imagen a un tamaño uniforme."""
    try:
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize(size)
        return img
    except Exception as e:
        raise RuntimeError(f"Error procesando imagen: {e}") from e


def obtener_embedding(imagen):
    """Genera el embedding de una imagen."""
    inputs = processor(images=imagen, return_tensors="pt")
    outputs = model.get_image_features(**inputs)

    return outputs / outputs.norm(dim=-1, keepdim=True)
