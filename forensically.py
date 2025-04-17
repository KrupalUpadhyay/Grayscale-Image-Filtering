import cv2
import numpy as np

def luminance_gradient(image_path, ksize=3, intensity=2.0, opacity=0.6):
    """Compute luminance gradient with adjustable intensity and opacity."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute Sobel gradients in X and Y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute magnitude of gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enhance contrast with intensity factor
    magnitude = cv2.convertScaleAbs(magnitude, alpha=intensity)

    # Apply a custom colormap for depth effect
    color_mapped = cv2.applyColorMap(magnitude, cv2.COLORMAP_TURBO)

    # Blend with the original image based on opacity
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image_colored, (1 - opacity), color_mapped, opacity, 0)

    return blended

image_path = "images\crop-2.png"
gradient_colored = luminance_gradient(image_path, ksize=31, intensity=5.0, opacity=2.0)
cv2.imwrite("luminance_gradient_depth.png", gradient_colored)
