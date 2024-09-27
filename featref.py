import cv2
import pandas as pd
import os
from threshold import process_image_threshold  # Importing from segmentation file
from featextrac import compute_features  # Importing feature extraction

# Function to generate and save reference features for given images
def generate_reference_features():
    # Dictionary to hold image paths and their labels
    image_paths = {
        "orange": "refimage.jpeg",
        "green_apple": "refimage2.jpeg"
    }
    
    feature_datasets = {}

    for label, image_path in image_paths.items():
        # Define the output CSV file name for each label
        output_csv = f"{label}_features_output.csv"
        
        if os.path.exists(output_csv):
            # If the CSV already exists, load the features
            print(f"Loading existing features for {label} from {output_csv}")
            feature_datasets[label] = pd.read_csv(output_csv, converters={"Color Histogram": eval})
        else:
            # Process the image if features have not been saved previously
            print(f"Processing and storing features for {label}")
            original_image, thresholded_image = process_image_threshold(image_path)
            
            if original_image is None or thresholded_image is None:
                print(f"Error: Image {image_path} not found or thresholding failed.")
                continue

            # Extract features using the feature extraction function
            feature_dataframe = compute_features(original_image, thresholded_image)
            
            # Save extracted features to CSV
            feature_dataframe.to_csv(output_csv, index=False)
            feature_datasets[label] = feature_dataframe

    return feature_datasets

# Main execution block
if __name__ == "__main__":
    reference_features = generate_reference_features()
    print("Reference features have been created and saved.")
