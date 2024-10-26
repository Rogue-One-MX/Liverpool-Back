"""Main Flask application."""

from flask import Flask, request, jsonify
from dotenv import load_dotenv
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
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image_data = file.read()

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
