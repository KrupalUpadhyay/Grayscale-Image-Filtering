import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Forensically output image
image_path = r"C:\Users\Krupal Upadhyay\OneDrive\Pictures\Screenshots\Screenshot 2025-03-03 111439.png"  # Change this to your file path
image = cv2.imread(image_path)

# Convert to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize for easier processing
resized = cv2.resize(image_rgb, (500, 100))

# Sample key color points along the intensity gradient
height, width, _ = resized.shape
sample_positions = np.linspace(0, width - 1, num=6, dtype=int)  # 6 key colors
sampled_colors = [resized[height // 2, x].tolist() for x in sample_positions]

# Display extracted colors
plt.figure(figsize=(8, 2))
for i, color in enumerate(sampled_colors):
    plt.bar(i, 1, color=np.array(color) / 255.0, width=1)
plt.xticks([])
plt.yticks([])
plt.title("Extracted Key Colors from Forensically Output")
plt.show()

# Print RGB values
print("Extracted RGB Colors:", sampled_colors)
