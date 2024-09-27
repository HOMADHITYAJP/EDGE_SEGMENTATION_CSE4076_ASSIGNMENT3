import cv2
import numpy as np
import pandas as pd

def compute_features(original_image, binary_image):
    detected_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    feature_data = {
        "Shape Area": [],
        "Shape Perimeter": [],
        "Bounding Rectangle": [],
        "Shape Centroid": [],
        "Color Histogram": []
    }

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    for contour in detected_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
        else:
            centroid_x, centroid_y = 0, 0


        contour_mask = np.zeros(gray_image.shape, np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        masked_image = cv2.bitwise_and(original_image, original_image, mask=contour_mask)
        color_histogram = cv2.calcHist([masked_image], [0, 1, 2], contour_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        normalized_histogram = cv2.normalize(color_histogram, color_histogram).flatten()

        feature_data["Shape Area"].append(area)
        feature_data["Shape Perimeter"].append(perimeter)
        feature_data["Bounding Rectangle"].append((x, y, width, height))
        feature_data["Shape Centroid"].append((centroid_x, centroid_y))
        feature_data["Color Histogram"].append(normalized_histogram.tolist())

    return pd.DataFrame(feature_data)

if __name__ == "__main__":
    input_image_path = "refimage.jpeg"
    binary_image_path = "thresholded_output.jpg"
    
    loaded_image = cv2.imread(input_image_path)
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    
    if loaded_image is None or binary_image is None:
        print("Error: Required image or binary image not found.")
    else:
        features_dataframe = compute_features(loaded_image, binary_image)
        features_dataframe.to_csv("extracted_features.csv", index=False)
        print("Features saved as extracted_features.csv")
