import keras_ocr

# Initialize the pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Specify the image path (replace with your actual image path)
image_path = 'test2.png'

# Read the image
image = keras_ocr.tools.read(image_path)

# Get the predictions
predictions = pipeline.recognize([image])  # Wrap image in a list

# Extract text and combine into a single string
text = ""
for word, _ in predictions[0]:  # Access first element from predictions
  text += word + " "  # Add space between words

# Print the extracted text
print(text.strip())  # Remove leading/trailing spaces