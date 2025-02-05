from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///annotations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

IMAGES_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images"))

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

    return jsonify({"image_id": image.id, "image_url": f"http://127.0.0.1:5000/api/image/{image.file_name}"})

@app.route('/api/image/<path:file_name>', methods=['GET'])
def serve_image(file_name):
    return send_from_directory(IMAGES_FOLDER, file_name)

@app.route('/api/submit', methods=['POST'])
def submit_annotation():
    data = request.get_json()
    print("Received data:", data) 

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
    annotations = Annotation.query.all()
    return jsonify([{ "worker_id": a.worker_id, "image_id": a.image_id, "segmentation_mask": a.segmentation_mask } for a in annotations])

if __name__ == '__main__':
    app.run(debug=True)
