import cv2
import numpy as np
import os
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# Function to create custom colormap
def create_custom_colormap():
    """Generates a fixed colormap for batch processing."""
    colors = [
        (255, 255, 255),  # White
        (40, 0, 80),      # Dark Purple
        (100, 0, 160),    # Purple
        (144, 0, 208),    # Magenta
        (192, 0, 255),    # Light Purple
        (255, 255, 255)   # White
    ]
    cmap = LinearSegmentedColormap.from_list("CUSTOM_MAP", colors, N=256)
    color_array = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    LUT = np.zeros((256, 1, 3), dtype=np.uint8)
    LUT[:, 0, :] = color_array
    return LUT

# Function to apply luminance gradient with inversion
def apply_luminance_gradient(image, ksize, intensity, opacity, colormap):
    """Applies the luminance gradient effect to an image with inversion."""

    # Ensure image is RGB (remove alpha if present)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Convert to grayscale and invert
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)  # 🔄 **Invert grayscale**

    # Compute Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize and enhance
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude = cv2.convertScaleAbs(magnitude, alpha=intensity)

    # Convert grayscale to 3-channel
    magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    # Apply custom colormap
    color_mapped = cv2.LUT(magnitude_bgr, colormap)

    # Ensure both images are the SAME size before blending
    color_mapped = cv2.resize(color_mapped, (image.shape[1], image.shape[0]))

    # Blend safely
    blended = cv2.addWeighted(image, (1 - opacity), color_mapped, opacity, 0)

    return blended

# Batch processing function
def process_folder(input_folder, output_folder, ksize=5, intensity=2.0, opacity=0.6):
    """Processes all images in a folder and saves results in another folder."""

    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load colormap once
    custom_lut = create_custom_colormap()

    # Process all images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping {filename} (Invalid image)")
                continue

            # Apply processing
            processed_image = apply_luminance_gradient(image, ksize, intensity, opacity, custom_lut)

            # Save result
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed & saved: {output_path}")

# Example usage
input_folder = r"C:\Users\Krupal Upadhyay\Downloads\LTEH\Radiography\dataset\images"  # Folder containing original images
output_folder = r"C:\Users\Krupal Upadhyay\Downloads\LTEH\Radiography\dataset\processed_images"  # Folder where processed images will be saved

# Set parameter values (Change if needed)
ksize = 31
intensity = 5.0
opacity = 0.9

process_folder(input_folder, output_folder, ksize, intensity, opacity)
print("✅ Batch processing complete!")
