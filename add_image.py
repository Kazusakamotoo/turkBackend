import os
from backend import db, Image, IMAGES_FOLDER, app

def add_images_to_db():
    with app.app_context():  # ✅ Ensure the Flask app context is active
        for file_name in os.listdir(IMAGES_FOLDER):
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                if not Image.query.filter_by(file_name=file_name).first():
                    db.session.add(Image(file_name=file_name))
        db.session.commit()
        print("✅ Images added to the database!")

if __name__ == "__main__":
    add_images_to_db()
