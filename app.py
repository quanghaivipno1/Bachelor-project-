from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import logging
import re

app = Flask(__name__)

# Load the models
model_4 = load_model('model_4.keras')
model_bin = load_model('model_bin.keras')
efficient_model = load_model('efficient_model.keras')

# Setup logging
logging.basicConfig(level=logging.INFO)

# Ensure the images directory exists
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)

# List of allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    # Extract the actual filename from the full path
    return os.path.basename(filename)

def pad_to_square(image):
    old_size = image.shape  # Get the current shape of the image
    desired_size = max(old_size)  # Determine the size of the square to pad to
    
    # Calculate padding widths
    padding_height = (desired_size - old_size[0]) // 2
    padding_width = (desired_size - old_size[1]) // 2
    
    # Determine padding for top, bottom, left, and right
    pad_top = padding_height
    pad_bottom = desired_size - old_size[0] - pad_top
    pad_left = padding_width
    pad_right = desired_size - old_size[1] - pad_left
    
    if len(image.shape) == 2:  # Grayscale image
        # Pad the image with constant value 110
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif len(image.shape) == 3:  # Color image
        # Pad the image with constant value 110 for each channel
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Unsupported image shape. Expected 2D grayscale or 3D color image.")
    
    return padded_image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        if 'imagefile' not in request.files and 'folderUpload' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        imagefiles = request.files.getlist('imagefile') + request.files.getlist('folderUpload')
        
        if len(imagefiles) == 0:
            return jsonify({'error': 'No selected files'}), 400
        
        model_type = request.form['model_type']
        
        results = []
        
        for imagefile in imagefiles:
            if sanitize_filename(imagefile.filename) == '' or not allowed_file(sanitize_filename(imagefile.filename)):
                continue
            
            sanitized_filename = sanitize_filename(sanitize_filename(imagefile.filename))
            print(sanitized_filename)
            image_path = os.path.join(images_dir, sanitized_filename)
            imagefile.save(image_path)
            
            # Load the image
            image = load_img(image_path,target_size=(40,24))
            image_array = img_to_array(image)
            print(image_array.shape)
            # if image_array.shape[-1] == 1:
            #     image_array = np.repeat(image_array, 3, axis=-1)

            # Prepare results dictionary for this image
            image_result = {'filename': sanitize_filename(imagefile.filename)}

            if model_type == 'model_4':
                # Model 1: Categorical classification
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
                gray_image = np.expand_dims(gray_image, axis=-1)
                gray_image = np.expand_dims(gray_image, axis=0)
                predictions_cat = model_4.predict(gray_image)[0]
                class_names = ['Offline-Module','Diode-Multi','Diode','Shadowing','Cell-Multi','Cell','Hot-Spot','Cracking','Hot-Spot-Multi','Soiling','Vegetation','No-Anomaly'] 
                image_result['Categorical Classification'] = class_names[np.argmax(predictions_cat)]

            elif model_type == 'model_bin':
                # Model 2: Binary classification (grayscale)
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.shape[-1] == 3 else image_array
                gray_image = np.expand_dims(gray_image, axis=-1)
                gray_image = np.expand_dims(gray_image, axis=0)
                print(gray_image.shape)
                predictions_bin = model_bin.predict(gray_image)[0]
                image_result['Binary Classification (Gray)'] = {
                    'Defective': f"{predictions_bin[0] * 100:.2f}%",
                    'Functional': f"{predictions_bin[1] * 100:.2f}%"
                }

            elif model_type == 'efficient_model':
                # Model 3: Binary classification (color)
                image_array=pad_to_square(image_array)
                image_array=np.expand_dims(image_array,axis=0)
                print(image_array.shape)
                predictions_bin_rgb = efficient_model.predict(image_array)[0]
                image_result['Binary Classification (Color)'] = {
                    'Defective': f"{predictions_bin_rgb[0] * 100:.2f}%",
                    'Functional': f"{predictions_bin_rgb[1] * 100:.2f}%"
                }

            # Apply color map and resize for display
            color_mapped_image = cv2.applyColorMap(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
            enlarged_image = cv2.resize(color_mapped_image, (240, 400), interpolation=cv2.INTER_CUBIC)
            enlarged_image_path = os.path.join(images_dir, 'enlarged_' + sanitize_filename(imagefile.filename))
            cv2.imwrite(enlarged_image_path, enlarged_image)
            
            image_result['enlarged_image'] = 'enlarged_' + sanitize_filename(imagefile.filename)
            results.append(image_result)

        print(f"Model type: {model_type}")
        print(f"Results: {results}")
        return render_template('index.html', results=results)

    except Exception as e:
        logging.error(f"Error processing images: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<filename>')
def send_image(filename):
    logging.info(f"Requested image: {filename}")
    return send_from_directory(images_dir, filename)

if __name__ == '__main__':
    os.environ["PYTHONIOENCODING"] = "utf-8"
    app.run(port=3000, debug=True)