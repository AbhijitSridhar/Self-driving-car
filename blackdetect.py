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
#This next part works only if edges are being detected. It turns off immediately if no edges are seen
#    rho = 2
#    theta = np.pi/180
#    threshold = 20
#    min_line_len = 50
#    max_line_gap = 200
#    lines = cv2.HoughLinesP(canny_edges,rho,theta,threshold,np.array([]),minLineLength = 50, maxLineGap = 200)
#    line_img = np.zeros((canny_edges.shape[0],canny_edges.shape[1],3),dtype = np.uint8)

#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(line_img,(x1,y1),(x2,y2),[255,0,0],2)

#    cv2.imshow('edges',line_img)
    cv2.imshow('out',canny_edges)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
