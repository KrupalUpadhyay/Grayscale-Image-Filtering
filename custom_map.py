import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormap(name="CUSTOM", colors=None, N=256):
    """
    Creates a custom colormap that closely resembles Forensically’s luminance gradient.
    
    Parameters:
        name (str): Name of the colormap.
        colors (list): List of RGB tuples defining the colormap.
        N (int): Number of color levels.
    
    Returns:
        LUT (np.ndarray): Lookup table for OpenCV.
    """
    if colors is None:
        colors = [
            (0, 30, 0),        # Deep Black
            (0, 40, 80),      # Dark Purple
            (10, 100, 150),    # Purple
            (50, 160, 210),   # Violet
            (120, 220, 250),  # Light Blue
            (255, 255, 255)   # White
        ]
    
    cmap = LinearSegmentedColormap.from_list(name, colors, N=N)

    # Convert to OpenCV format (256 x 1 x 3)
    color_array = (cmap(np.linspace(0, 1, N))[:, :3] * 255).astype(np.uint8)
    LUT = np.zeros((256, 1, 3), dtype=np.uint8)
    LUT[:, 0, :] = color_array

    return LUT

def luminance_gradient(image_path, ksize=5, intensity=3.5, opacity=0.8, colormap=None):
    """Compute luminance gradient with enhanced intensity, opacity, and accurate colormap matching."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute Sobel gradients in X and Y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute magnitude of gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enhance contrast with intensity factor
    magnitude = cv2.convertScaleAbs(magnitude, alpha=intensity)

    # Apply a custom colormap if provided, else use TURBO
    if colormap is not None:
        color_mapped = cv2.LUT(cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR), colormap)
    else:
        color_mapped = cv2.applyColorMap(magnitude, cv2.COLORMAP_TURBO)

    # Ensure opacity does not exceed 1.0
    opacity = min(opacity, 1.0)

    # Blend with the original image
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image_colored, (1 - opacity), color_mapped, opacity, 0)

    return blended

# Load Image and Apply Luminance Gradient with Custom Colormap
image_path = "images/crop-2.png"

# Generate custom colormap LUT matching Forensically
custom_lut = create_custom_colormap(name="MY_COLORMAP")

# Apply luminance gradient with adjusted parameters
gradient_colored = luminance_gradient(image_path, ksize=7, intensity=3.0, opacity=0.9, colormap=custom_lut)

# Save results
cv2.imwrite("luminance_gradient_forensically.png", gradient_colored)
