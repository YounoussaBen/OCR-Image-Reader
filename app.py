from flask import Flask, request, jsonify
import keras_ocr

# Initialize Flask app
app = Flask(__name__)

# Initialize the keras_ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Define POST endpoint for text extraction
@app.route('/extract_text', methods=['POST'])
def extract_text():
    # Check if request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # Check if file is an image
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Unsupported file format. Only PNG, JPG, and JPEG are supported.'}), 400
    
    # Read the image
    image = keras_ocr.tools.read(file)
    
    # Get the predictions
    predictions = pipeline.recognize([image])  # Wrap image in a list
    
    # Extract text and combine into a single string
    text = ""
    for word, _ in predictions[0]:  # Access first element from predictions
        text += word + " "  # Add space between words
    
    return jsonify({'output': text.strip()}), 200

if __name__ == '__main__':
    app.run(debug=False)
