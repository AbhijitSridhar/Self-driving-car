import cv2
import matplotlib.pyplot as plt
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()

#conversion to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#conversion to hsv
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#applying masks to extract red and green 
    red_lower = np.array([0,50,50], dtype = np.uint8)
    red_upper = np.array([10,255,255], dtype = np.uint8)
    mask_red = cv2.inRange(hsv_image, red_lower, red_upper)
    mask_final_r = cv2.bitwise_and(gray_image, mask_red)

    green_lower = np.array([40,100,100], dtype =np.uint8)
    green_upper = np.array([80,255,255], dtype = np.uint8)
    mask_green = cv2.inRange(hsv_image, green_lower, green_upper)
    mask_final_g = cv2.bitwise_and(gray_image, mask_green)


#THE NEXT SECTION IS FOR DETECTING CIRCLES, CURRENTLY NOT BEING USED

#    blur_image = cv2.medianBlur(gray_image, 5)
#    circle_image = cv2.cvtColor(blur_image, cv2.COLOR_GRAY2BGR)

#    circles = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 30)

#    circles = np.uint16(np.around(circles))

#    for i in circles[0, :]:
#        cv2.circle(circle_image, (i[0],i[1]),i[2], (0,255,0), 2)
#        cv2.circle(circle_image,(i[0],i[1]),2,(0,0,255),3)

#END OF CIRCLE DETECTION

#Determining wether ouput is red or green and writing Foo accordingly
    red_black = cv2.countNonZero(mask_red)
    if red_black > 2000:
        print('RED')
        Foo = 0

    green_black = cv2.countNonZero(mask_final_g)
    if green_black > 1800:
        print('GREEN')
        Foo = 1

#   cv2.imshow('detected circles', circle_image)
#   cv2.imshow('gray',gray_image)
    cv2.imshow('red',mask_final_r)
    cv2.imshow('green', mask_final_g)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()

