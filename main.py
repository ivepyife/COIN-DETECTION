#Import all the library
import cv2
import numpy as np
import matplotlib.pyplot as plt


image   = cv2.imread('assets/koin5.jpg') #import image                                                                
resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) #halving the image size and use interpolation technique              
gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #transform image to grayscale                                             
blur    = cv2.GaussianBlur(gray, (11, 11), 0) #blur the image using GaussianBlur technique                                                    
canny   = cv2.Canny(blur, 30, 150) #using canny technique                                                              
dilated = cv2.dilate(canny, (1,1), iterations=2) #dilated the image                                                

#finding the contour value
(cnt, _) = cv2.findContours(                                                                     
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

circles = [] #container for result
for contour in cnt:
    area = cv2.contourArea(contour) #get area
    perimeter = cv2.arcLength(contour, True) #get perimeter
    circularity = 4 * np.pi * area / (perimeter * perimeter) #circularity formula
    if circularity > 0.8:
        circles.append(contour)

rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).copy()
cv2.drawContours(rgb, circles, -1, (0, 255, 0), 2)
#show the amount of detected coin
print("Number of coins detected:", len(circles))

#show the process
cv2.imshow('resized', resized)
cv2.imshow('grayscale', gray)
cv2.imshow('blurred', blur)
cv2.imshow('cannied', canny)
cv2.imshow('dilated', dilated)

#show the result
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title('Original Picture')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(rgb)
plt.title('After')
plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()                                                                        
