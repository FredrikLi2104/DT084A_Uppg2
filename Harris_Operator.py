# The Harris Operator in this code has been written with the help of chatGPT
import numpy as np
import cv2
import matplotlib.pyplot as plt

def sift_detector():
    img = cv2.imread('img1.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    
    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imwrite('olle.jpg',img)
    
    cv2.imshow('SIFT Keypoints',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# k is constant between 0.04 and 0.06
def harris_corner_detector(img, k=0.04, threshold=0.2):
    # Convert to grayscale since Harris detects corners in grayscale pictures
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # Harris Corner takes float as input
    gray = np.float32(gray)

    # Compute image gradients, ksize is a parameter which determines the size
    # of the Sobel kernel 3x3, 5x5 etc as size increases more pixels are part
    # of each convolution process and the edges will get more blurry.
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of derivatives based on the matrix |Ix*Ix Ix*Iy|
    #                                                     |Ix*Iy Iy*Iy|
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Apply Gaussian filter to smooth
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 1)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 1)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 1)

    # Compute response function R, detA is determinant, k is constant
    # R = det(M) - c*trace(M)^2 = lam1*lam2 - c(lam1 + lam2)^2
    detA = (Ixx * Iyy) - (Ixy * Ixy)
    traceA = Ixx + Iyy
    R = detA - k * (traceA ** 2)

    # Normalize R values
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    

    # Apply threshold to detect strong corners
    corners = R > threshold

    return corners, R

# Load and test on an image
img = cv2.imread('img1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
corners, response = harris_corner_detector(img, k=0.04, threshold=0.29)

img2 = cv2.imread('img2.jpg')
img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
corners2, response2 = harris_corner_detector(img2, k=0.04, threshold=0.2)
# Visualize result
# Show response heatmap
plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.imshow(response, cmap='jet')
plt.title('Harris Response1')

# Show detected corners
plt.subplot(2, 2, 2)
plt.imshow(img_rgb)
y, x = np.where(corners > 0)
plt.scatter(x, y, s=10, c='red', label='Corners')
plt.legend()
plt.title('Harris Corners1')

plt.subplot(2, 2, 3)
plt.imshow(response2, cmap='jet')
plt.title('Harris Response2')

# Show detected corners
plt.subplot(2, 2, 4)
plt.imshow(img_rgb2)
y, x = np.where(corners2 > 0)
plt.scatter(x, y, s=10, c='red', label='Corners')
plt.legend()
plt.title('Harris Corners2')
plt.show()

sift_detector()
