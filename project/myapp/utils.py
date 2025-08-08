# utils.py

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img_eq = cv2.equalizeHist(img)
    g_kernel = cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F)
    img_gabor = cv2.filter2D(img_eq, cv2.CV_8UC3, g_kernel)

    try:
        thresh_val = threshold_otsu(img_gabor)
        binary = img_gabor > thresh_val
    except Exception:
        binary = img_gabor > img_gabor.mean()

    binary = (binary.astype(np.uint8) * 255)
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
