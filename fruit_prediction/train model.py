import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

# Extract color, texture, and shape features from an image
def extract_features(image):
    image = cv2.resize(image, (100, 100))  # Resize to make all images uniform
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray_flat = gray.flatten()  # Flatten the grayscale image into 1D array

    # Convert to HSV color space and extract color histogram (texture + color info)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Shape features using threshold and contours
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize shape features
    area = 0
    perimeter = 0
    aspect_ratio = 0
    extent = 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0

    shape_features = np.array([area, perimeter, aspect_ratio, extent])

    return np.hstack((gray_flat, hist, shape_features))

# Load images and labels from dataset folder
def load_dataset(dataset_dir='/home/shrutika/Desktop/ippr/dataset'):
    features = []
    labels = []
    for label_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, label_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    feat = extract_features(image)
                    features.append(feat)
                    labels.append(label_folder)
    return np.array(features), np.array(labels)

# Train the model and save it
def train_model():
    X, y = load_dataset()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    joblib.dump(model, 'fruit_seed_classifier.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')
    print("✅ Model and encoder saved using color, size, shape and texture features.")

# Run training
if __name__ == "__main__":
    train_model()
