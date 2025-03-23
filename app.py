import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  # For YOLOv8
import tensorflow as tf  # For TensorFlow/Keras

# Define the disease classes
disease_classes = [
    "Stem_canker",
    "fruit_Phytopthora",
    "fruit_Scab",
    "fruit_Styler end Root",
    "fruit_healthy",
    "leaf_Anthracnose",
    "leaf_Canker",
    "leaf_Mummification",
    "leaf_Red rust",
    "leaf_Rust",
    "leaf_healthy",
    "stem_healthy",
    "stem_wilt",
]

# Define YOLO class labels
yolo_labels = ['Guava Fruit', 'Guava Leaf', 'Guava Stem']

# Disease prevention tips
disease_prevention = {
    "Stem_canker": (
        "Fungal infection",
        "Apply copper-based fungicides and prune infected stems."
    ),
    "fruit_Phytopthora": (
        "Waterlogged soil",
        "Improve drainage and avoid overwatering."
    ),
    "fruit_Scab": (
        "Fungal infection",
        "Use fungicides and remove infected fruits."
    ),
    "fruit_Styler end Root": (
        "Root damage",
        "Ensure proper irrigation and avoid root injuries."
    ),
    "fruit_healthy": (
        "No issue detected",
        "Maintain regular care and monitoring."
    ),
    "leaf_Anthracnose": (
        "Fungal infection",
        "Prune affected leaves and apply fungicides."
    ),
    "leaf_Canker": (
        "Bacterial infection",
        "Remove infected parts and sanitize tools."
    ),
    "leaf_Mummification": (
        "Nutrient deficiency",
        "Apply balanced fertilizers."
    ),
    "leaf_Red rust": (
        "Cephaleuros virescens (algal pathogen), High humidity and poor plant health",
        "Spray Bordeaux mixture (1%), Prune affected leaves, Avoid excessive watering."
    ),
    "leaf_Rust": (
        "Puccinia psidii fungus, High humidity and prolonged leaf wetness",
        "Use sulfur-based fungicides, Avoid overcrowding of plants, Improve air circulation."
    ),
    "leaf_healthy": (
        "No issue detected",
        "Maintain regular care and monitoring."
    ),
    "stem_healthy": (
        "No issue detected",
        "Maintain regular care and monitoring."
    ),
    "stem_wilt": (
        "Fusarium oxysporum fungus, Poor drainage and excessive moisture, Root damage due to nematodes",
        "Improve soil drainage, Apply Trichoderma-based biofungicides, Remove and destroy infected plants."
    ),
}

# Load YOLOv8s model for object detection
def load_yolo_model():
    return YOLO('models/best.pt')  # Pretrained YOLOv8s model

# Load DenseNet-201 model for classification (TensorFlow/Keras)
def load_keras_model():
    return tf.keras.models.load_model('models/fine_epoch_64.h5')

# Define preprocessing for DenseNet-201 (TensorFlow/Keras)
def preprocess_image(image):
    # Resize and normalize the image for DenseNet-201
    image = image.resize((224, 224))  # Resize to 224x224
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to perform object detection using YOLOv8
def detect_objects(yolo_model, image):
    # Perform inference with YOLOv8
    results = yolo_model(image)  # Run inference
    detections = results[0].boxes.data.cpu().numpy()  # Get detections as numpy array
    return detections

# Function to classify a cropped region using DenseNet-201 (TensorFlow/Keras)
def classify_image(keras_model, cropped_image):
    # Preprocess the cropped image
    img_tensor = preprocess_image(cropped_image)
    # Perform inference with Keras model
    predictions = keras_model.predict(img_tensor)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class
    return predicted_class

# Main pipeline function
def detection_classification_pipeline(yolo_model, keras_model, image_path):
    # Load the input image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Object detection with YOLOv8
    detections = detect_objects(yolo_model, image_rgb)

    results = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf < 0.5:  # Filter out low-confidence detections
            continue

        # Crop the detected region
        cropped_region = image[int(y1):int(y2), int(x1):int(x2)]
        if cropped_region.size == 0:  # Skip invalid crops
            continue
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))

        # Step 2: Classification with DenseNet-201 (TensorFlow/Keras)
        class_id = classify_image(keras_model, cropped_pil)

        # Map YOLO class to disease class
        detected_object = yolo_labels[int(cls)]
        disease_class = disease_classes[class_id]

        # Append results
        results.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'detected_object': detected_object,
            'disease_class': disease_class
        })

    return results

# Save results to a JSON file
def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Main function
def main():
    # Load models
    yolo_model = load_yolo_model()
    keras_model = load_keras_model()

    # Directory containing test images
    test_images_dir = "test_images"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the test directory
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        print(f"Processing {image_path}...")

        # Run the pipeline
        results = detection_classification_pipeline(yolo_model, keras_model, image_path)

        # Save results to a JSON file
        output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_results.json")
        save_results(results, output_file)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
