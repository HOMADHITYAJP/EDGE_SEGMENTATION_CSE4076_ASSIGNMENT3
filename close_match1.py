# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:56:26 2024

@author: homap
"""

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
from featref import generate_reference_features

# Function to compute normalized color histograms for circular regions
def compute_circular_region_histogram(image, center_coords, radius):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(center_coords[0]), int(center_coords[1])), int(radius), 255, -1)
    color_hist = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    normalized_hist = cv2.normalize(color_hist, color_hist).flatten()
    return normalized_hist

# Function to compute the maximum Euclidean distance between any two histograms
def compute_max_distance(reference_data):
    max_distance = 0
    for category, features_df in reference_data.items():
        for idx, row1 in features_df.iterrows():
            hist1 = np.array(row1["Color Histogram"])
            for idx2, row2 in features_df.iterrows():
                hist2 = np.array(row2["Color Histogram"])
                dist = distance.euclidean(hist1, hist2)
                if dist > max_distance:
                    max_distance = dist
    return max_distance

# Function to find the closest match between target and reference histograms
def close_match(reference_data, target_hist, max_distance):
    best_match = None
    lowest_distance = float('inf')

    for category, features_df in reference_data.items():
        for idx, row in features_df.iterrows():
            reference_hist = np.array(row["Color Histogram"])
            dist = distance.euclidean(reference_hist, target_hist)

            if dist < lowest_distance:
                lowest_distance = dist
                best_match = category

    # Normalize the distance
    normalized_score = lowest_distance / max_distance
    return best_match, normalized_score

if __name__ == "__main__":
    # Load pre-generated reference feature data
    ref_feature_data = generate_reference_features()

    # Compute the maximum possible distance for normalization
    max_dist = compute_max_distance(ref_feature_data)

    # Load target image for feature comparison
    target_image_path = "image.jpeg"
    target_image = cv2.imread(target_image_path)

    if target_image is None:
        print(f"Error: Image {target_image_path} not found.")
    else:
        regions_of_interest = []
        while True:
            # Select ROI using OpenCV's built-in selector
            roi_bbox = cv2.selectROI("Select Region", target_image, fromCenter=False, showCrosshair=True)

            if roi_bbox == (0, 0, 0, 0):
                break

            regions_of_interest.append(roi_bbox)

        for idx, roi_bbox in enumerate(regions_of_interest):
            x, y, w, h = roi_bbox
            selected_region = target_image[y:y+h, x:x+w]
            gray_region = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)
            _, binary_roi = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Extract contours within the thresholded region
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (circle_x, circle_y), circle_radius = cv2.minEnclosingCircle(largest_contour)

                adjusted_center = (int(circle_x + x), int(circle_y + y))
                circle_radius = int(circle_radius)

                # Compute histogram for the circular region
                target_hist = compute_circular_region_histogram(target_image, adjusted_center, circle_radius)

                # Match target histogram with reference data
                label, match_score = close_match(ref_feature_data, target_hist, max_dist)

                # Draw circle and centroid
                cv2.circle(target_image, adjusted_center, circle_radius, (0, 255, 0), 3)
                cv2.circle(target_image, adjusted_center, 5, (0, 0, 255), -1)

                label_position = (adjusted_center[0] - circle_radius, max(adjusted_center[1] - circle_radius - 20, 20))
                label_text = f"{label} ({match_score:.2f})"

                # Customize font size and style
                cv2.putText(target_image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save and display final result
        output_image_path = "final_output.jpg"
        cv2.imwrite(output_image_path, target_image)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        plt.title("Identified Regions with Labels")
        plt.axis('off')
        plt.show()

        print(f"Labeled image saved as {output_image_path}")
