import cv2
import numpy as np

coins = cv2.imread("/Users/jinheppell/Desktop/Coding/open cv/Detecting circles/coins.jpg")
grey_coins = cv2.cvtColor(coins,cv2.COLOR_BGR2GRAY)
median_blur_coins = cv2.medianBlur(grey_coins,5)
detected_circles = cv2.HoughCircles(median_blur_coins, cv2.HOUGH_GRADIENT ,1 ,20, param1 = 50,param2 = 30, minRadius = 1, maxRadius = 40)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :] :
        a,b,r = pt[0],pt[1],pt[2]
        cv2.circle(coins, (a,b), r, (0, 255, 0), 2)
        cv2.circle(coins, (a, b), 1, (0, 0, 255), 3)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100

params.filterByCircularity = True
params.minCircularity = 0.9

params.filterByConvexity = True
params.minConvexity = 0.1

params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(coins)

blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(coins, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

cv2.imshow("Filtering Circular Blobs Only",blobs)

cv2.waitKey(0)
cv2.destroyAllWindows()