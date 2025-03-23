import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
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

# Disease prevention tips (updated with detailed information)
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
@st.cache_resource
def load_yolo_model():
    return YOLO('C:\\Users\\rlaha\\OneDrive\\Desktop\\app\\guava-disease-detection\\models\\best.pt')  # Pretrained YOLOv8s model

# Load DenseNet-201 model for classification (TensorFlow/Keras)
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('C:\\Users\\rlaha\\OneDrive\\Desktop\\app\\guava-disease-detection\\models\\fine_epoch_64.h5')

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

# Draw bounding boxes on the image
def draw_bounding_boxes(image, results):
    draw = ImageDraw.Draw(image)
    for result in results:
        bbox = result['bbox']
        label = f"{result['detected_object']} ({result['disease_class']})"
        confidence = result['confidence']
        draw.rectangle(bbox, outline="lime", width=3)
        draw.text((bbox[0], bbox[1] - 15), f"{label} {confidence:.2f}", fill="lime")
    return image

# Sidebar: Index and Upload Section
def sidebar_index():
    st.sidebar.title("Guava Farm Doctor 🍃")
    st.sidebar.markdown("### Uploaded Images")
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []

    # Display uploaded images in the sidebar
    for i, img in enumerate(st.session_state.uploaded_images[:5]):
        st.sidebar.image(img, caption=f"Image {i+1}", use_column_width=True)

    # Upload new image
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Add to session state
        img = Image.open(uploaded_file)
        if len(st.session_state.uploaded_images) >= 5:
            st.session_state.uploaded_images.pop(0)  # Remove oldest image
        st.session_state.uploaded_images.append(img)

        # Return the path of the uploaded image
        return image_path
    return None

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
            'bbox': (x1, y1, x2, y2),
            'confidence': conf,
            'detected_object': detected_object,
            'disease_class': disease_class
        })

    return results, image_rgb

# Streamlit UI
def main():
    # Title
    st.title("Guava Farm Doctor 🍃")

    # Sidebar: Index and Upload Section
    image_path = sidebar_index()

    if image_path:
        # Load models
        yolo_model = load_yolo_model()
        keras_model = load_keras_model()

        # Process the image
        with st.spinner("Processing image..."):
            results, image_rgb = detection_classification_pipeline(yolo_model, keras_model, image_path)

        # Two-column layout
        col1, col2 = st.columns([2, 1])

        # Column 1: Annotated image and classification results
        with col1:
            st.subheader("Annotated Image")
            annotated_image = Image.fromarray(image_rgb)
            annotated_image = draw_bounding_boxes(annotated_image, results)
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

            # Display classification results
            st.subheader("Classification Results")
            for result in results:
                disease_class = result['disease_class']
                detected_object = result['detected_object']

                # Replace "disease" with "healthy" if the classification is healthy
                if "healthy" in disease_class.lower():
                    display_disease_class = disease_class.replace("healthy", "Healthy")
                    st.success(f"Object: {detected_object} | Status: {display_disease_class} | Confidence: {result['confidence']:.2f}")
                else:
                    st.success(f"Object: {detected_object} | Disease: {disease_class} | Confidence: {result['confidence']:.2f}")

        # Column 2: Placeholder for additional information
        with col2:
            st.subheader("Additional Information")
            st.info("This section can include metadata, notes, or other details about the image.")

        # Full-width section: Disease prevention tips
        st.subheader("Prevention and Care Tips")
        data = []
        for result in results:
            disease = result['disease_class']
            reason, prevention = disease_prevention.get(disease, ("Unknown", "Consult an expert."))

            # Replace "disease" with "healthy" if the classification is healthy
            if "healthy" in disease.lower():
                disease = disease.replace("healthy", "Healthy")
                reason = "No issue detected"
                prevention = "Maintain regular care and monitoring."

            data.append({"Status": disease, "Cause": reason, "Prevention": prevention})

        # Display as a table
        if data:
            st.table(data)
        else:
            st.info("No issues detected.")

    else:
        st.info("Please upload an image to proceed.")

# Run the app
if __name__ == "__main__":
    main()