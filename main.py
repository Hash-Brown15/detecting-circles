import cv2
import numpy as np

sunset = cv2.imread("/Users/jinheppell/Desktop/Coding/open cv/Detecting circles/sunset.webp")
grey_sunset = cv2.cvtColor(sunset,cv2.COLOR_BGR2GRAY)
median_blur_sunset = cv2.medianBlur(grey_sunset,5)
detected_circles = cv2.HoughCircles(median_blur_sunset, cv2.HOUGH_GRADIENT ,1 ,20, param1 = 50,param2 = 30, minRadius = 1, maxRadius = 40)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :] :
        a,b,r = pt[0],pt[1],pt[2]
        cv2.circle(sunset, (a,b), r, (0, 255, 0), 2)
        cv2.circle(sunset, (a, b), 1, (0, 0, 255), 3)

cv2.imshow("Final Result",sunset)

cv2.waitKey(0)
cv2.destroyAllWindows()