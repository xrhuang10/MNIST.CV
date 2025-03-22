from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

camera = cv2.VideoCapture(0)  # Open webcam (0 for default camera)

def generate_frames():
    while True:
        success, frame = camera.read()  # Read a frame
        frame = cv2.flip(frame, 1)  # Mirror the frame
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)  # Draw blue rectangle

        ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Return frame data
            

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
