"""Main Flask application."""

from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from dotenv import load_dotenv
import requests
from ai import preprocesar_imagen, obtener_embedding, index

load_dotenv()

app = Flask(__name__)


@app.route("/")
def home():
    """Ruta de inicio."""
    return "Hello, Flask!"


@app.route("/find-similar", methods=["POST"])
def find_similar():
    """Encuentra la imagen más similar en el índice Pinecone."""
    # Check if the "image_url" field is in the JSON body
    data = request.get_json()
    if not data or "image_url" not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data["image_url"]

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download image: {e}"}), 500

    preprocessed_image = preprocesar_imagen(image_data)
    if not preprocessed_image:
        return jsonify({"error": "Image processing failed"}), 500

    uploaded_embedding = obtener_embedding(preprocessed_image)
    uploaded_embedding_flat = (
        uploaded_embedding.detach().cpu().numpy().flatten().tolist()
    )

    query_response = index.query(
        vector=uploaded_embedding_flat, top_k=1, include_metadata=True
    )

    if query_response["matches"]:
        most_similar = query_response["matches"][0]
        result = {
            "sku": most_similar["id"],
            "similarity_score": most_similar["score"],
        }
        return jsonify(result)
    else:
        return jsonify({"error": "No similar embeddings found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
