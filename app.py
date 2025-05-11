from flask import Flask, render_template, request
from flask import Response
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Make sure the folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model_path = r'model/model.h5'
model = load_model(model_path)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32)).convert('L')  # Resize + Grayscale
    img = np.array(img).reshape(32, 32, 1)
    img = np.repeat(img, 3, axis=2)  # Make it 3-channel if needed
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            img_tensor = preprocess_image(path)
            probs = model.predict(img_tensor)[0]
            top_index = np.argmax(probs)
            prediction = f"{class_names[top_index]} ({probs[top_index]*100:.2f}%)"
            
    return render_template('index.html', prediction=prediction, filename=filename)

@app.route('/submit', methods=['POST'])
def submit():
    prediction = None
    filename = None
    path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            img_tensor = preprocess_image(path)
            probs = model.predict(img_tensor)[0]
            top_index = np.argmax(probs)
            prediction = f"{class_names[top_index]} ({probs[top_index]*100:.2f}%)"

    return render_template('submit.html', title='Prediction Result', image_url=path, pred=class_names[top_index], desc=prediction)

@app.route('/contact-us', methods=['GET'])
def contact_us():
    return render_template('contact-us.html')

# Camera stream with live prediction
def camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for prediction
        img = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(32, 32, 1)
        gray = np.repeat(gray, 3, axis=2)
        gray = gray / 255.0
        img_tensor = np.expand_dims(gray, axis=0)

        # Predict
        probs = model.predict(img_tensor)[0]
        top_index = np.argmax(probs)
        label = f"{class_names[top_index]} ({probs[top_index]*100:.2f}%)"

        # Overlay prediction on frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/camera_feed')
def camera_feed():
    return Response(camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

# Route for live prediction page
@app.route('/live-predict')
def live_predict():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)
