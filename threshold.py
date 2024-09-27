import cv2

def process_image_threshold(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image, threshold_image

def extract_contours(threshold_image):
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_edges(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_image, 100, 200)


def draw_and_save_contours(image, contours, output_file):
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(output_file, contoured_image)

def save_and_display_results(threshold_image, contours, edges_image, original_image):
    cv2.imwrite("thresholded_output.jpg", threshold_image)
    print("Saved thresholded image as thresholded_output.jpg")
    
    draw_and_save_contours(original_image, contours, "contours_output.jpg")
    print("Saved contours image as contours_output.jpg")

    cv2.imwrite("edges_output.jpg", edges_image)
    print("Saved edges image as edges_output.jpg")

    cv2.imshow("Threshold", threshold_image)
    cv2.imshow("Contours", cv2.imread("contours_output.jpg"))
    cv2.imshow("Edges", edges_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution flow
image_file = "refimage.jpeg"
original_img, thresholded_img = process_image_threshold(image_file)
contour_results = extract_contours(thresholded_img)
edge_detected_image = detect_edges(original_img)

save_and_display_results(thresholded_img, contour_results, edge_detected_image, original_img)
