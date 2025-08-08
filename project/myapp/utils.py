import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


def estimate_orientation(img):
    # Compute gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation angle (in degrees)
    orientation = 0.5 * np.arctan2(2 * np.mean(sobelx * sobely), np.mean(sobelx**2 - sobely**2))
    angle = orientation * (180 / np.pi)
    return angle

def rotate_image(img, angle):
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

# In preprocess_fingerprint, after loading grayscale img:
angle = estimate_orientation(img)
img = rotate_image(img, -angle)  # Rotate to normalize orientation

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 1. Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    # 2. Denoise with fast Non-local Means Denoising
    img_denoised = cv2.fastNlMeansDenoising(img_eq, h=10)

    # 3. Edge-preserving smoothing with bilateral filter
    img_bilateral = cv2.bilateralFilter(img_denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Optional median blur to remove salt and pepper noise
    img_median = cv2.medianBlur(img_bilateral, 3)

    # 5. Apply Gabor filter to enhance ridge patterns
    g_kernel = cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F)
    img_gabor = cv2.filter2D(img_median, cv2.CV_8UC3, g_kernel)

    # 6. Thresholding - Try Otsu first, fallback to mean threshold
    try:
        thresh_val = threshold_otsu(img_gabor)
        binary = img_gabor > thresh_val
    except Exception:
        binary = img_gabor > img_gabor.mean()

    # 7. Morphological operations to clean small noise & fill gaps
    binary = (binary.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 8. Skeletonize ridges to 1-pixel width
    skeleton = skeletonize(binary // 255)
    skeleton = (skeleton * 255).astype(np.uint8)

    return skeleton
    
def match_fingerprint(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    if img1 is None or img2 is None:
        return 0

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_percent = len(good_matches) / len(matches) if matches else 0
    return match_percent
