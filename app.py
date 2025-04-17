import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# Function to create custom colormap
def create_custom_colormap(colors):
    """Generates a colormap from the user-defined colors."""
    cmap = LinearSegmentedColormap.from_list("CUSTOM_MAP", colors, N=256)
    color_array = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    LUT = np.zeros((256, 1, 3), dtype=np.uint8)
    LUT[:, 0, :] = color_array
    return LUT

# Function to apply luminance gradient with inversion
def apply_luminance_gradient(image, ksize, intensity, opacity, colormap):
    """Applies the luminance gradient effect to an image with inversion."""
    
    # Ensure the image is RGB (remove alpha if present)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
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

    # Ensure data types match
    color_mapped = color_mapped.astype(np.uint8)
    image = image.astype(np.uint8)

    # Blend safely
    blended = cv2.addWeighted(image, (1 - opacity), color_mapped, opacity, 0)

    return blended

# Streamlit UI
st.set_page_config(layout="wide")  # Enable wide layout
st.title("🔥 Photo Forensics Web App")
st.write("Adjust sliders to see real-time effects!")

# Layout Columns: Left (Images) | Right (Controls)
col1, col2 = st.columns([2, 1])

with col2:  # Right column: Sliders & Color Pickers
    st.subheader("🔧 Adjust Parameters")
    
    # UI Sliders
    ksize = st.slider("Kernel Size", 1, 31, 5, step=2)
    intensity = st.slider("Intensity", 0.1, 5.0, 2.0)
    opacity = st.slider("Opacity", 0.0, 1.0, 0.6)

    st.subheader("🎨 Customize Colormap")

    # Color Pickers Only
    colormap_colors = [
        st.color_picker("Color-1", "#FFFFFF"),
        st.color_picker("Color-2", "#280050"),
        st.color_picker("Color-3", "#6400A0"),
        st.color_picker("Color-4", "#9000D0"),
        st.color_picker("Color-5", "#C000FF"),
        st.color_picker("Color-6", "#FFFFFF"),
    ]

with col1:  # Left column: Image Upload & Display
    uploaded_file = st.file_uploader("📸 Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Generate colormap & apply effect
        custom_lut = create_custom_colormap(colormap_colors)
        processed_image = apply_luminance_gradient(image, ksize, intensity, opacity, custom_lut)

        # Increased Image Size for Preview (Set width to 650px)
        st.image([image, processed_image], caption=["Original Image", "Processed Image"], width=650)

        # Download Button
        processed_image_pil = Image.fromarray(processed_image)
        st.download_button("Download Processed Image", data=processed_image_pil.tobytes(), file_name="processed.png", mime="image/png")

st.write("🎨 Adjust sliders & colors for real-time effects. Upload any image and save the output!")
