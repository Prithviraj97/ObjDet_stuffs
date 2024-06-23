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



from flask import Flask, render_template, request, Response, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
# Create 'static' and 'uploaded_videos' directories if they don't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/uploaded_videos'):
    os.makedirs('static/uploaded_videos')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_person(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    persons = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in persons:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_person(frame)
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
            frame = detect_person(frame)
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
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)