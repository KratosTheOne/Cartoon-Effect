"""
Th following script is used to read the captured/selected image
and create a process of steps through which the image goes through
and gets converted into a cartoon form.
The script generates 3 images as outputs hence showing
the various stages through which the image is going through
namely blurred, edged out & cartoon image.
"""

import cv2
import numpy as np

# Reading the Image
input_image = cv2.imread("sample2.jpeg")

# Finding the Edges of Image
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)

# Making a Cartoon of the image
color = cv2.bilateralFilter(input_image, 12, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)

# Visualize the cartoon image
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)  # "0" is Used to close the image window
cv2.destroyAllWindows()

"""
Final Step: Convert to grayscale, apply gaussian blur
and blur images heavily with edgePreservingFilter
"""
# convert to gray scale
grayImage = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# apply gaussian blur
grayImage = cv2.GaussianBlur(grayImage, (3, 3), 0)

# detect edges
edgeImage = cv2.Laplacian(grayImage, -1, ksize=5)
edgeImage = 255 - edgeImage

# threshold image
ret, edgeImage = cv2.threshold(edgeImage, 150, 255, cv2.THRESH_BINARY)

# blur images heavily using edgePreservingFilter
edgePreservingImage = cv2.edgePreservingFilter(input_image, flags=2, sigma_s=50, sigma_r=0.4)

# create output matrix
output = np.zeros(grayImage.shape)

# combine cartoon image and edges image
output = cv2.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edgeImage)

# Visualize the cartoon image
cv2.imshow("Cartoon", output)
cv2.waitKey(0)  # "0" is Used to close the image window
cv2.destroyAllWindows()

cartoon_image = cv2.stylization(input_image, sigma_s=150, sigma_r=0.25)
cv2.imshow('cartoon', cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
