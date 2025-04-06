from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#USE MEDIAPIPE HAND DETECTOR INSTEAD

lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

camera = cv2.VideoCapture(0)  # Open webcam (0 for default camera)

prev_x, prev_y = None, None  # Reset on app start
canvas = None  # Delay initialization

def camera_on():
    global prev_x, prev_y, canvas
    
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))  # Resize for consistency
        if canvas is None:
            canvas = np.zeros_like(frame)  # Match canvas to frame shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                h, w, _ = frame.shape
                finger_tips = [8, 12]  # Index and middle fingertips

                fingers_up = []
                for tip in finger_tips:
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        fingers_up.append(True)
                    else:
                        fingers_up.append(False)

                if fingers_up == [True, True]:
                    cx = int(landmarks[8].x * w)
                    cy = int(landmarks[8].y * h)

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 4)
                    prev_x, prev_y = cx, cy
                else:
                    prev_x, prev_y = None, None

        combined = cv2.addWeighted(frame, 1, canvas, 0.6, 0)


        ret, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def save_image():
    img_path = "static/drawing.jpg"
    cv2.imwrite(img_path, canvas)
    print(f"Image saved to {img_path}")
    return img_path  # Return the path for use in the frontend


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera_on(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calculate', methods=['POST'])
def calculate():
    img_path = save_image()
    #return "This is the calculated page"
    return {'image_path': img_path}



if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
