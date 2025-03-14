from PIL import Image, ImageDraw
import json 

training_folder = './person_resume_funsd_format_v5/dataset/training'
annotation_folder = '/annotations/'
images_folder = '/images/'

json_file = training_folder + annotation_folder + "resume_1_0.json"
image_file = training_folder + images_folder + "resume_1_0.jpeg"

with open(json_file) as jf:
    json_data = json.load(jf)

data = {
    "objects":[
        json_data
    ],
    "imagePath": image_file
}

img = Image.open(data["imagePath"])
# img = img.resize((1900, 1200))        # resize - only for tests

# draw = ImageDraw.Draw(img)

# for item in data["form"]:
#     x1, y1, x2, y2 = item["box"]
#     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue boxes
#     cv2.putText(image, item["text"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    
# img.show()

import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Create a drawing context
draw = ImageDraw.Draw(img)

# Loop through the bounding boxes and draw them
for item in json_data["form"]:
    x1, y1, x2, y2 = item["box"]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1 - 10), item["label"], fill="red")

# Display the image with bounding boxes
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.axis("off")
plt.savefig("testing_resume_bounding_box_visual.pdf")
plt.show()
