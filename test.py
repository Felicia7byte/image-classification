from transformers import pipeline
from PIL import Image

classifier = pipeline("image-classification")

image = Image.open("image.jpg")

result = classifier(image)

print(result)