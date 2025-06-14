import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
from PIL import Image

# Load your trained FER model
model = load_model('model.h5', compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buffer size for smoothing predictions
SMOOTHING_WINDOW = 10

# Dictionary to hold prediction buffers for each detected face (by position)
prediction_buffers = {}

def preprocess_face(face_img):
    img = Image.fromarray(face_img)
    img_arr = np.array(img, dtype=np.float32)
    img_arr /= 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)  # (48,48,1)
    img_arr = np.expand_dims(img_arr, axis=0)   # (1,48,48,1)
    return img_arr

def get_smoothed_prediction(buffer):
    # Average probabilities in buffer, then argmax
    avg_preds = np.mean(buffer, axis=0)
    return np.argmax(avg_preds)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        new_prediction_buffers = {}

        for i, (x, y, w, h) in enumerate(faces):
            pad = 10
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, gray.shape[1])
            y2 = min(y + h + pad, gray.shape[0])

            face_roi = gray[y1:y2, x1:x2]

            # Histogram equalization
            face_roi = cv2.equalizeHist(face_roi)

            face_resized = cv2.resize(face_roi, (48, 48))

            processed = preprocess_face(face_resized)

            preds = model.predict(processed)[0]

            # Create a key for this face based on position (rounded to avoid jitter)
            face_key = (round(x1 / 10), round(y1 / 10), round(w / 10), round(h / 10))

            # Get existing buffer or create new
            buffer = prediction_buffers.get(face_key, deque(maxlen=SMOOTHING_WINDOW))
            buffer.append(preds)
            new_prediction_buffers[face_key] = buffer

            emotion_idx = get_smoothed_prediction(buffer)
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)

        prediction_buffers.clear()
        prediction_buffers.update(new_prediction_buffers)

        cv2.imshow('Emotion Detection - Press q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
