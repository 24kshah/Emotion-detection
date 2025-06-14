import os
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained FER model (assumed to be in project root)
model = load_model('model.h5', compile=False)

# Emotion labels in FER order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_and_crop_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (48, 48))
    return face_img

def preprocess_fer_image(face_img):
    # face_img is a numpy array of shape (48,48), grayscale
    img = Image.fromarray(face_img)
    img_arr = np.array(img, dtype=np.float32)
    img_arr /= 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)  # (48,48,1)
    img_arr = np.expand_dims(img_arr, axis=0)   # (1,48,48,1)
    return img_arr

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        face_img = detect_and_crop_face(filepath)
        if face_img is None:
            return render_template('result.html', emotion="No face detected. Please upload a clear face image.")

        processed_img = preprocess_fer_image(face_img)
        preds = model.predict(processed_img)
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]

        return render_template('result.html', emotion=emotion)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
