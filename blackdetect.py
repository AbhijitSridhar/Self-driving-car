import cv2
import matplotlib.pyplot as plt
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(gray_image,0,50)
    mask_image = cv2.bitwise_and(gray_image,mask_black)


    kernel_size = 5
    gauss_gray = cv2.GaussianBlur(mask_image,(kernel_size,kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(gauss_gray,low_threshold,high_threshold)

    cv2.imshow('edges',canny_edges)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
