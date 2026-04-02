from flask import Flask, render_template, request
import os
from utils.face_detection import detect_face
from utils.predict import predict

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    faces = detect_face(filepath)

    if len(faces) == 0:
        return "No face detected"

    face = faces[0]

    label, confidence = predict(face)

    confidence = round(confidence * 100, 2)

    return render_template(
        'result.html',
        label=label,
        confidence=confidence
    )


if __name__ == '__main__':
    app.run(debug=True)