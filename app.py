from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import json
import os
import base64
import google.generativeai as genai
import cv2

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///annotations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

IMAGES_FOLDER = "image"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False, unique=True)

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    worker_id = db.Column(db.String(50), nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    segmentation_mask = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route('/api/image', methods=['GET'])
def get_random_image():
    image = Image.query.order_by(db.func.random()).first()
    if not image:
        return jsonify({"error": "No images found in database"}), 404

    BASE_URL = "https://turkbackendai.onrender.com" 

    return jsonify({"image_id": image.id, "image_url": f"{BASE_URL}/api/image/{image.file_name}"})


@app.route('/api/image/<path:filename>', methods=['GET'])
def serve_image(filename):
    file_path = os.path.join(IMAGES_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"File {filename} not found in {IMAGES_FOLDER}"}), 404
    return send_from_directory(IMAGES_FOLDER, filename)

def encode_image(image_path, bbox):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found or cannot be read: {image_path}")

    if not bbox or len(bbox) != 4:
        raise ValueError(f"Invalid bounding box: {bbox}")

    x, y, width, height = map(int, bbox) 
    x2, y2 = x + width, y + height 

    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 3)

    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def verify_bbox_with_gemini(image_path, bbox):
    encoded_image = encode_image(image_path, bbox)

    prompt = """
    This image contains an animal with a user-drawn bounding box. 
    Please determine:
    1. If the bounding box fully encloses the animal.
    2. If the bounding box is too small, too large, or well-fitted.
    3. Provide a short explanation.
    
    Return your answer as "Valid" or "Invalid" followed by a reason.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            prompt,
            {"mime_type": "image/jpeg", "data": encoded_image},
        ]
    )

    return response.text

@app.route('/api/validate', methods=['POST'])
def validate_annotation():
    """Validates a bounding box using Gemini API and ensures JSON responses."""
    data = request.get_json()

    print("Received validation request:", json.dumps(data, indent=2))

    if not data or "image_id" not in data or "bounding_box" not in data:
        print("❌ Invalid request format!")
        return jsonify({"error": "Invalid request format"}), 400

    bbox = data["bounding_box"]

    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(i, (int, float)) for i in bbox):
        print("❌ Invalid bounding box format:", bbox)
        return jsonify({"error": "Bounding box must be a list of 4 numbers"}), 400

    image = Image.query.get(data["image_id"])
    if not image:
        return jsonify({"error": "Image not found"}), 404

    image_path = os.path.join(IMAGES_FOLDER, image.file_name)

    try:
        gemini_response = verify_bbox_with_gemini(image_path, bbox)

        return jsonify({
            "valid": "Valid" in gemini_response,
            "reason": gemini_response
        })

    except ValueError as ve:
        print(f"Value Error: {str(ve)}") 
        return jsonify({"error": "Value Error", "details": str(ve)}), 400

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route('/api/submit', methods=['POST'])
def submit_annotation():
    data = request.get_json()

    if not data or 'worker_id' not in data or 'annotations' not in data:
        return jsonify({"error": "Invalid request"}), 400

    if not data["annotations"]:
        return jsonify({"error": "No annotations received"}), 400

    for annotation in data["annotations"]:
        if "image_id" not in annotation or "bounding_boxes" not in annotation:
            return jsonify({"error": "Invalid annotation format"}), 400

        new_annotation = Annotation(
            worker_id=data["worker_id"],
            image_id=annotation["image_id"],
            segmentation_mask=json.dumps(annotation["bounding_boxes"])
        )
        db.session.add(new_annotation)

    db.session.commit()
    return jsonify({"message": "Annotations submitted successfully"})

@app.route('/api/annotations', methods=['GET'])
def get_annotations():
    """Returns all stored annotations."""
    annotations = Annotation.query.all()
    return jsonify([
        {
            "id": a.id,
            "worker_id": a.worker_id,
            "image_id": a.image_id,
            "bounding_boxes": json.loads(a.segmentation_mask),
            "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        } for a in annotations
    ])

@app.route('/debug/files', methods=['GET'])
def list_files():
    """Lists all images in the image directory for debugging."""
    if not os.path.exists(IMAGES_FOLDER):
        return jsonify({"error": "Image folder not found"}), 404
    files = os.listdir(IMAGES_FOLDER)
    return jsonify({"files": files})

if __name__ == '__main__':
    app.run(debug=True)
