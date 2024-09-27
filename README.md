# EDGE_SEGMENTATION_CSE4076_ASSIGNMENT3

# Object Identification Assignment

## Overview
This repository contains an assignment focused on object identification using image processing techniques with OpenCV and Python. The objective is to identify specific objects (oranges and green apples) in images by applying segmentation, feature extraction, and matching algorithms.

## Directory Structure
├── close_match1.py # Script for identifying objects based on color histograms 
├── featref.py # Script for generating reference features from images 
├── featextrac.py # Script for extracting features from images 
├── threshold.py # Script for image segmentation using thresholding 
├── images/ # Directory containing images used in the assignment 
├── image.jpeg # Target image for object identification 
├── refimage.jpeg # Reference image of an orange 
├── refimage2.jpeg # Reference image of a green apple 
├── README.md # Documentation for the assignment

## Installation
To run this assignment, ensure you have Python installed on your system along with the necessary libraries. You can install the required libraries using pip:

BASH: 
pip install opencv-python numpy pandas matplotlib scipy.

Usage

1.	Image Segmentation: Run threshold.py to preprocess the images and create binary masks for the objects of interest.
python threshold.py
2.	Feature Extraction: Use featextrac.py to extract features from the preprocessed images.
BASH
python featextrac.py
3.	Reference Feature Generation: Execute featref.py to generate and save reference features for known objects (oranges and green apples).
BASH
python featref.py
4.	Object Identification: Finally, run close_match1.py to identify and label the objects in the target image.
BASH
python close_match1.py

Outputs

The output images generated during the process are as follows:

•	Thresholded Image: Displays the binary mask where objects are separated from the background.

•	Identified Objects: Shows the identified oranges and green apples in the target image, marked with circles and labeled accordingly.

Conclusion

This assignment demonstrates the application of image processing techniques for the identification of specific objects in images. The methods used include segmentation, feature extraction, and histogram comparison to effectively classify and recognize oranges and green apples.

Key Takeaways

•	Understanding the image processing pipeline from segmentation to feature extraction and classification.
•	Gaining hands-on experience with OpenCV and Python for real-world applications in computer vision.
•	Insights into the challenges and solutions associated with object identification.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

Author

•	HOMADHITYA J P

Feel free to modify the text to better fit your style or add any additional information that might be relevant to your assignment.

