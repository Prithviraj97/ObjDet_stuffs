# from flask import Flask, render_template, request, Response
# import cv2
# import numpy as np

# app = Flask(__name__)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# def detect_person(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     persons = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in persons:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return frame

# def gen_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame = detect_person(frame)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen_frames_webcam():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame = detect_person(frame)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed', methods=['POST'])
# def video_feed():
#     video_path = request.form.get('video_path')
#     if video_path:
#         return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/webcam_feed')
# def webcam_feed():
#     return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     video_file = request.files['file']
#     video_path = 'uploaded_videos/' + video_file.filename
#     video_file.save(video_path)
#     return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, request, Response, url_for
# import cv2
# import numpy as np
# import os

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
# # Create 'static' and 'uploaded_videos' directories if they don't exist
# if not os.path.exists('static'):
#     os.makedirs('static')
# if not os.path.exists('static/uploaded_videos'):
#     os.makedirs('static/uploaded_videos')

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_person(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     persons = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in persons:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return frame

# def gen_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame = detect_person(frame)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen_frames_webcam():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame = detect_person(frame)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     video_path = request.args.get('video_path')
#     if video_path:
#         return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/webcam_feed')
# def webcam_feed():
#     return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     video_file = request.files['video_file']
#     video_path = os.path.join('static/uploaded_videos', video_file.filename)
#     print(video_path)  # Print the video_path to see where the file is being saved
#     video_file.save(video_path)
#     return 'Video uploaded successfully'

# @app.route('/uploaded_video_feed')
# def uploaded_video_feed():
#     video_path = request.args.get('video_path')
#     return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

    ###New CODE HERE
# from flask import Flask, render_template, request, Response, url_for
# import cv2
# import numpy as np
# import os

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
# # Create 'static' and 'uploaded_videos' directories if they don't exist
# if not os.path.exists('static'):
#     os.makedirs('static')
# if not os.path.exists('static/uploaded_videos'):
#     os.makedirs('static/uploaded_videos')

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_person(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     persons = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in persons:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return frame

# def gen_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame = detect_person(frame)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen_frames_webcam():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frame = detect_person(frame)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     video_path = request.args.get('video_path')
#     if video_path:
#         return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/webcam_feed')
# def webcam_feed():
#     return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     video_file = request.files['video_file']
#     video_path = os.path.join('static/uploaded_videos', video_file.filename)
#     print(video_path)  # Print the video_path to see where the file is being saved
#     video_file.save(video_path)
#     return 'Video uploaded successfully'

# @app.route('/uploaded_video_feed')
# def uploaded_video_feed():
#     video_path = request.args.get('video_path')
#     video_path = os.path.join('static', video_path)  # Prepend 'static/' to the video_path
#     return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

##Code to include YOLO
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import os
import torch
from PIL import Image
from sort import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create 'static' and 'uploaded_videos' directories if they don't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/uploaded_videos'):
    os.makedirs('static/uploaded_videos')

# Initialize SORT tracker
mot_tracker = Sort()

def detect_person(frame):
    results = model(frame)
    persons = results.xyxy[0].numpy()
    return persons

def track_persons(persons):
    dets = []
    for person in persons:
        x, y, x2, y2, _, _ = person.astype(int)
        dets.append([x, y, x2, y2, 1.0])
    dets = np.array(dets)
    tracks = mot_tracker.update(dets)
    return tracks

def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            persons = detect_person(frame)
            tracks = track_persons(persons)
            for track in tracks:
                x, y, x2, y2, _ = track.astype(int)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{track[4]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            persons = detect_person(frame)
            tracks = track_persons(persons)
            for track in tracks:
                x, y, x2, y2, _ = track.astype(int)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{track[4]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    if video_path:
        return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['video_file']
    video_path = os.path.join('static/uploaded_videos', video_file.filename)
    print(video_path)  # Print the video_path to see where the file is being saved
    video_file.save(video_path)
    return 'Video uploaded successfully'

@app.route('/uploaded_video_feed')
def uploaded_video_feed():
    video_path = request.args.get('video_path')
    video_path = os.path.join('static', video_path)  # Prepend 'static/' to the video_path
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)