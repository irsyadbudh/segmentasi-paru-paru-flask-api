from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict_image
from PIL import Image

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the 'uploads' folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define allowed extensions
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_dimensions(filepath):
    with Image.open(filepath) as img:
        return img.size

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = predict_image(filepath)
        image_dimensions = get_image_dimensions(filepath)
        return render_template('prediction.html', result=result, image_path=filename, image_dimensions=image_dimensions)
    else:
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
