import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('black1.jpg')
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('img',gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask_black = cv2.inRange(gray_image,0,150)
mask_image = cv2.bitwise_and(gray_image,mask_black)


kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_image,(kernel_size,kernel_size),0)
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray,low_threshold,high_threshold)
plt.imshow(canny_edges)
plt.show()
