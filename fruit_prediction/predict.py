import cv2
import numpy as np
import joblib
import os

def extract_features(image):
    image = cv2.resize(image, (100, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_flat = gray.flatten()

    # HSV Color Histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Binary image for shape analysis
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = 0
    perimeter = 0
    aspect_ratio = 0
    extent = 0

    if contours:
        c = max(contours, key=cv2.contourArea)  # largest contour
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0

    shape_features = np.array([area, perimeter, aspect_ratio, extent])

    # Combine all features
    return np.hstack((gray_flat, hist, shape_features))

def load_model_and_encoder():
    model = joblib.load('fruit_seed_classifier.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

def predict_seed_type(image_name):
    image_directory = './input'
    image_path = os.path.join(image_directory, image_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Image '{image_name}' not found.")
        return

    features = extract_features(image).reshape(1, -1)
    model, encoder = load_model_and_encoder()

    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform(prediction)[0]

    print(f"\n✅ The seed type is: {predicted_label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        print("\n🔍 Prediction confidence:")
        for label, prob in zip(encoder.classes_, probabilities):
            print(f"{label}: {prob:.2f}")

    # Annotate and show image
    display_image = image.copy()
    cv2.putText(display_image, f'Predicted: {predicted_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Seed Prediction", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_name = input("Enter the name of the image to classify (e.g., 'test_seed.jpg'): ")
    predict_seed_type(image_name)