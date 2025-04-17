import cv2
import numpy as np

# ---------------------- MODULES ----------------------

def apply_clahe(image):
    """Enhance contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_laplacian(image):
    """Apply Laplacian Edge Detection."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def apply_canny(image, low_thresh=50, high_thresh=150):
    """Apply Canny Edge Detection."""
    return cv2.Canny(image, low_thresh, high_thresh)

def combine_edges(edge1, edge2, weight1=0.5, weight2=0.5):
    """Combine two edge detection results."""
    return cv2.addWeighted(edge1, weight1, edge2, weight2, 0)

def apply_morphological_closing(image, kernel_size=3):
    """Apply Morphological Closing to fill gaps."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_adaptive_threshold(image):
    """Apply Adaptive Thresholding for segmentation."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

# ---------------------- PIPELINE ----------------------

# Load grayscale image
image = cv2.imread(r"C:\Users\Krupal Upadhyay\OneDrive\Pictures\Screenshots\radiography.png", cv2.IMREAD_GRAYSCALE)

# Define your custom pipeline sequence
step1 = apply_clahe(image)
final_output = apply_adaptive_threshold(step1)
step2 = apply_laplacian(final_output)  
step3 = apply_canny(step2)  
# step4 = combine_edges(step2, step3)  
# step5 = apply_morphological_closing(step4)  
  

# Save only the final output
cv2.imwrite("processed.jpg", final_output)

print("Processing complete. Final image saved as 'final_processed_image.jpg'.")
