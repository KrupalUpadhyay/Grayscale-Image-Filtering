import cv2
import numpy as np

def load_image(image_path):
    """Load the grayscale radiography image."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def apply_gaussian_blur(image, kernel_size=5):
    """Apply Gaussian Blur to reduce noise while preserving edges."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_blur(image, kernel_size=5):
    """Apply Median Blur to remove salt-and-pepper noise."""
    return cv2.medianBlur(image, kernel_size)

def apply_sobel_edge_detection(image):
    """Apply Sobel filter in both X and Y directions."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_scharr_edge_detection(image):
    """Apply Scharr filter for stronger edge detection."""
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    return cv2.normalize(scharr_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_prewitt_edge_detection(image):
    """Apply Prewitt filter for edge detection."""
    prewitt_x = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.float32)  # Convert to float32
    return cv2.normalize(prewitt_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_clahe(original):
    """Enhance contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(original)

# 🚀 **Modify the image path as needed**
image_path = "results\overlay-laplacian.png"
original = load_image(image_path)

# Apply different filters (Uncomment the ones you want)
clahe = apply_clahe(original)
gaussian = apply_gaussian_blur(clahe)
median = apply_median_blur(gaussian)
sobel = apply_sobel_edge_detection(median)
scharr = apply_scharr_edge_detection(sobel)
prewitt = apply_prewitt_edge_detection(scharr)

# Save processed images
cv2.imwrite("gaussian_blur.png", gaussian)
cv2.imwrite("median_blur.png", median)
cv2.imwrite("sobel_edges.png", sobel)
cv2.imwrite("scharr_edges.png", scharr)
cv2.imwrite("prewitt_edges.png", prewitt)

print("Processing complete. Check the output images.")
