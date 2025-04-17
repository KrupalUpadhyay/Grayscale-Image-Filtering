import cv2
import numpy as np

print(cv2.getBuildInformation())


def load_image(image_path):
    """Load the grayscale radiography image and normalize to 0-255."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imwrite("normalized_check.png", img)
    return img

def threshold_belt_region(image):
    """Apply adaptive thresholding to highlight belt regions."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 125, 4)

def apply_laplacian(image, restricted=True):
    """Apply Laplacian filter to enhance edges. If restricted, normalize intensities."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX) if restricted else laplacian

def apply_canny(image, restricted=True):
    """Apply Canny edge detection. If restricted, normalize intensities."""
    edges = cv2.Canny(image, 30, 120)
    return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX) if restricted else edges

def overlay_edges(original, edges):
    """Overlay detected edges onto the original image for better visualization."""
    return cv2.addWeighted(original, 0.7, edges, 0.3, 0)

def apply_clahe(original):
    """Enhance contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(original)

def apply_morphological_operations(image):
    """Apply morphological operations for noise reduction and edge enhancement."""
    kernel = np.ones((5,5), np.uint8)  # 5x5 kernel for transformations
    
    erosion = cv2.erode(image, kernel, iterations=1)  # Shrinks bright regions (removes noise)
    # dilation = cv2.dilate(image, kernel, iterations=1)  # Expands bright regions (fills gaps)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Fills small holes
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)  # Removes small noise
    dilation = cv2.dilate(opening, kernel, iterations=1)  # Expands bright regions (fills gaps)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)  # Highlights edges

    return erosion, dilation, opening, closing, gradient

def apply_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Apply bilateral filtering to reduce noise while preserving edges.
    
    Parameters:
        d (int): Diameter of the pixel neighborhood.
        sigmaColor (float): Filter sigma in color space.
        sigmaSpace (float): Filter sigma in coordinate space.
    
    Returns:
        Filtered image.
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def apply_contour_filtering(image, min_area=245):
    """
    Filter out small contours based on area to keep only significant defects.
    
    Parameters:
        min_area (int): Minimum contour area to keep.
    
    Returns:
        Processed image with only significant defects.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros_like(image)  # Create an empty mask
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:  # Keep only significant defects
            cv2.drawContours(filtered_image, [cnt], -1, 255, thickness=-1)

    return filtered_image

# Load image
image_path = "images\crop-1.png"  # Change this to your file path
original = load_image(image_path)
print(original.dtype, original.min(), original.max())

# Apply transformations
bilateral = apply_bilateral_filter(original)
clahe = apply_clahe(original)
thresholded = threshold_belt_region(clahe)
laplacian = apply_laplacian(thresholded)
# canny = apply_canny(laplacian)
overlay_laplacian = overlay_edges(thresholded, laplacian)
# overlay_canny = overlay_edges(overlay_laplacian, canny)

# Apply morphological operations
erosion, dilation, opening, closing, gradient = apply_morphological_operations(overlay_laplacian)
# filtered_defects = apply_contour_filtering(original)

# Save processed images
cv2.imwrite("results\check-results\crop-1.png", dilation)
# cv2.imwrite("erosion.png",erosion)
# cv2.imwrite("dilation.png",dilation)
# cv2.imwrite("opening.png",opening)
# cv2.imwrite("closing.png",closing)
# cv2.imwrite("gradient.png",gradient)
print("Processing complete. Processed images saved.")