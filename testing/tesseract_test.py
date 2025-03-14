import pytesseract
import cv2
from resume_parser import resumeparse
import resume_parser

# data = resumeparse.read_file('/resume.txt')

# print(data)

print(resume_parser.__file__)



pytesseract.pytesseract.tesseract_cmd = r"E:\Program files\tesseract-ocr\tesseract.exe"

img_cv = cv2.imread(r"./testing/High-School-Student-Resume-Sample.png")

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img_cv))

resume_text = pytesseract.image_to_string(img_cv)

# print(resume_text)

text_file = 'resume.txt'


# with open(text_file, '+w') as f:
#     f.write(resume_text)





# print(pytesseract.image_to_string(image="./testing/image1.jpg", lang="eng"))

