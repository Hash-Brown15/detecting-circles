import cv2
import numpy as np

castle = cv2.imread("/Users/jinheppell/Desktop/Coding/Coding Homework/detecting circles Hw/castle.jpg")
greyscale_castle = cv2.cvtColor(castle,cv2.COLOR_BGR2GRAY)
gaussian_blur = cv2.GaussianBlur(greyscale_castle,(7,7),0)
detected_circles = cv2.HoughCircles(gaussian_blur, cv2.HOUGH_GRADIENT,20 ,20)
cv2.imshow("Final Result",detected_circles)

cv2.waitKey(0)
cv2.destroyAllWindows()