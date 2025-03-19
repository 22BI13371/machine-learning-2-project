import cv2
import numpy as np
import os
import pytesseract
import json

# File paths (adjust as needed)
image_path = "person_resume_funsd_format_v5/dataset/processed_data/images/0a763412-resume_3067_0.jpeg"
annotation_path = "person_resume_funsd_format_v5/dataset/processed_data/labels/0a763412-resume_3067_0.txt"  # YOLO annotation file

# Example mapping from YOLO class id to label name
label_map = {
    "0": "Experience",
    "1": "Personal_Info",
    "2": "Skills"
}

if not os.path.exists('cropped_images'):
    os.mkdir('cropped_images')

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise Exception("Image not found: " + image_path)

img_height, img_width = img.shape[:2]

def yolo_to_pixels(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    """
    Convert normalized YOLO coordinates to pixel values.
    Returns: (x_min, y_min, width, height)
    """
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    box_width = width_norm * img_width
    box_height = height_norm * img_height
    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    return x_min, y_min, int(box_width), int(box_height)

# Dictionary to collect cropped images by label
cropped_by_label = {}

# Read YOLO annotations
with open(annotation_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        continue  # Skip invalid lines

    class_id, x_center_norm, y_center_norm, width_norm, height_norm = parts
    x_center_norm = float(x_center_norm)
    y_center_norm = float(y_center_norm)
    width_norm = float(width_norm)
    height_norm = float(height_norm)
    
    # Convert YOLO normalized coordinates to pixels
    x_min, y_min, w, h = yolo_to_pixels(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height)
    
    # Ensure bounding box is within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    w = min(w, img_width - x_min)
    h = min(h, img_height - y_min)
    
    # Crop the region from the image
    cropped = img[y_min:y_min+h, x_min:x_min+w]
    
    # Map class id to a label (default to "unknown" if not found)
    label = label_map.get(class_id, "unknown")
    
    # Store cropped image in our dictionary
    if label not in cropped_by_label:
        cropped_by_label[label] = []
    cropped_by_label[label].append(cropped)


# to hold final json object
extracted_text_dict = {}

# For each label, combine all cropped regions into one image and save it.
for label, crops in cropped_by_label.items():
    # Determine the maximum width and the total height for a vertical concatenation.
    max_width = max(crop.shape[1] for crop in crops)
    total_height = sum(crop.shape[0] for crop in crops)
    
    # Create a blank white canvas
    output_img = 255 * np.ones((total_height, max_width, 3), dtype=np.uint8)
    
    current_y = 0
    for crop in crops:
        h, w = crop.shape[:2]
        # If needed, pad the crop to match the maximum width
        if w < max_width:
            pad = 255 * np.ones((h, max_width - w, 3), dtype=np.uint8)
            crop = np.concatenate((crop, pad), axis=1)
        # Place the crop into the output image
        output_img[current_y:current_y+h, 0:max_width] = crop
        current_y += h

    # Extract text using Tesseract OCR
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    custom_config = r'--oem 3 --psm 6'  # Adjust config as needed
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
    
    extracted_text_dict[label] = extracted_text
    
    # Save the output image for this label
    output_filename = f"cropped_images/{label}_output.png"
    cv2.imwrite(output_filename, output_img)
    print(f"Saved output for label '{label}' as {output_filename}")
    
print(len(cropped_by_label))
# print(extracted_text_dict)
with open('test.json', 'w+') as f:
    json.dump(extracted_text_dict, f)