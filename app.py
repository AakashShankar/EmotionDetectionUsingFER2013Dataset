# from flask import Flask, render_template, request, Response
# import cv2
# import numpy as np
# from keras.models import model_from_json
# from werkzeug.utils import secure_filename
# import os
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#
# # Load the model
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#
# # Load JSON and create model
# with open("D:\\tenserFlowProject\\PublishMEemotionDet\\model\\emotion_model.json", 'r') as json_file:
#     loaded_model_json = json_file.read()
#
# emotion_model = model_from_json(loaded_model_json)
# emotion_model.load_weights("D:\\tenserFlowProject\\PublishMEemotionDet\\model\\emotion_model.h5")
# print("Model loaded successfully")
#
# # Preprocessing function
# def preprocess_frame(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     resized_frame = cv2.resize(gray_frame, (48, 48))
#     normalized_frame = resized_frame / 255.0
#     reshaped_frame = np.expand_dims(normalized_frame, axis=-1)  # Add channel dimension
#     reshaped_frame = np.expand_dims(reshaped_frame, axis=0)  # Add batch dimension
#     return reshaped_frame
#
# # Predict emotion from a frame
# def predict_emotion(frame):
#     processed_frame = preprocess_frame(frame)
#     prediction = emotion_model.predict(processed_frame)
#     max_index = int(np.argmax(prediction))
#     return emotion_dict[max_index]
#
# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return "No file uploaded", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No file selected", 400
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#
#     # If image, predict immediately
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image = cv2.imread(filepath)
#         emotion = predict_emotion(image)
#         return render_template('result.html', result=emotion, media_type="image", file_path=filepath)
#
#     # If video, process it and stream results
#     elif filename.lower().endswith('.mp4'):
#         return Response(process_video(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(process_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# def process_video(filepath):
#     cap = cv2.VideoCapture(filepath)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         emotion = predict_emotion(frame)
#         cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     cap.release()
#
# def process_webcam():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         emotion = predict_emotion(frame)
#         cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     cap.release()
#
# @app.route('/stop')
# def stop():
#     return render_template('index.html')
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from keras.models import model_from_json
from werkzeug.utils import secure_filename
import os

# app = Flask(__name__)
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, template_folder='templates', static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

with open("D:\\tenserFlowProject\\PublishMEemotionDet\\model\\emotion_model.json", 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("D:\\tenserFlowProject\\PublishMEemotionDet\\model\\emotion_model.h5")
print("Model loaded successfully")

# Face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return gray_frame, faces

# Predict emotion
def predict_emotion(face):
    resized_face = cv2.resize(face, (48, 48)) / 255.0
    reshaped_face = np.expand_dims(np.expand_dims(resized_face, axis=-1), axis=0)
    prediction = emotion_model.predict(reshaped_face)
    max_index = int(np.argmax(prediction))
    return emotion_dict[max_index]

# Draw results on frame
def draw_results(frame, faces, emotions):
    for (x, y, w, h), emotion in zip(faces, emotions):
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process uploaded image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(filepath)
        gray_frame, faces = preprocess_frame(frame)
        emotions = [predict_emotion(gray_frame[y:y + h, x:x + w]) for (x, y, w, h) in faces]
        draw_results(frame, faces, emotions)
        cv2.imwrite(filepath, frame)
        return render_template('result.html', result="Detection Complete", media_type="image", file_path=filepath)

    # Process uploaded video
    elif filename.lower().endswith('.mp4'):
        return Response(process_video(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(process_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame, faces = preprocess_frame(frame)
        emotions = [predict_emotion(gray_frame[y:y + h, x:x + w]) for (x, y, w, h) in faces]
        draw_results(frame, faces, emotions)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

def process_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame, faces = preprocess_frame(frame)
        emotions = [predict_emotion(gray_frame[y:y + h, x:x + w]) for (x, y, w, h) in faces]
        draw_results(frame, faces, emotions)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/stop')
def stop():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
