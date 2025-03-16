import cv2
import pytesseract

# File paths (adjust as needed)
image_path = "person_resume_funsd_format_v5/dataset/processed_data/images/0a763412-resume_3067_0.jpeg"
annotation_path = "person_resume_funsd_format_v5/dataset/processed_data/labels/0a763412-resume_3067_0.txt"  # YOLO annotation file

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise Exception("Image not found at " + image_path)
img_height, img_width, _ = image.shape

def yolo_to_pixels(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    """
    Convert normalized YOLO coordinates to pixel values.
    """
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    box_width = width_norm * img_width
    box_height = height_norm * img_height
    # Calculate top-left corner (x_min, y_min)
    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    return x_min, y_min, int(box_width), int(box_height)

# Container for extracted text
final_text = []

# Read YOLO annotations
with open(annotation_path, "r") as f:
    lines = f.readlines()

for line in lines:
    # YOLO format: class_id x_center y_center width height
    parts = line.strip().split()
    if len(parts) != 5:
        continue  # Skip invalid lines
    
    # Parse annotation values
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = parts
    x_center_norm = float(x_center_norm)
    y_center_norm = float(y_center_norm)
    width_norm = float(width_norm)
    height_norm = float(height_norm)
    
    # Convert normalized coordinates to pixel values
    x_min, y_min, w, h = yolo_to_pixels(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height)
    
    # Ensure coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    w = min(w, img_width - x_min)
    h = min(h, img_height - y_min)
    
    # Crop the image region
    cropped_region = image[y_min:y_min+h, x_min:x_min+w]
    
    # Optional: Preprocess cropped_region if needed (e.g., grayscale, threshold)
    # cropped_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    
    # Extract text using Tesseract OCR
    custom_config = r'--oem 3 --psm 6'  # Adjust config as needed
    extracted_text = pytesseract.image_to_string(cropped_region, config=custom_config)
    
    if extracted_text.strip():
        final_text.append(extracted_text.strip())

# Save the extracted text to a file
with open("extracted_text.txt", "w") as out_file:
    out_file.write("\n".join(final_text))

print("Extraction complete. Check 'extracted_text.txt'.")