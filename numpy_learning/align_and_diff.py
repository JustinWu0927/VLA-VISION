import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def align_and_diff(master_path, test_path):
    # 1. Grayscale
    master = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if master is None or test is None:
        print("Error: Could not load images. Check your file paths.")
        return

    # 2. Extract SIFT features and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(master, None)
    kp2, des2 = sift.detectAndCompute(test, None)

    # 3. Match features using FLANN
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # 4. Filter for robust matches using Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 5. Calculate Homography
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the 3x3 matrix that warps the test image to the master image
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        height, width = master.shape
        aligned_test = cv2.warpPerspective(test, H, (width, height))
    else:
        print("Not enough matches found to align images.")
        return

    # Find contours of the detected defect regions
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_test = cv2.cvtColor(aligned_test, cv2.COLOR_GRAY2BGR)
    total_defect_area = 0
    
    # Filter noise: only flag blobs larger than 20 pixels
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            total_defect_area += area
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_test, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Engineering Pass/Fail threshold (e.g., max allowable defect area = 50 sq px)
    MAX_ALLOWED_DEFECT_AREA = 50
    verdict = "PASS" if total_defect_area <= MAX_ALLOWED_DEFECT_AREA else "FAIL"
    status_color = (0, 255, 0) if verdict == "PASS" else (0, 0, 255)

    cv2.putText(annotated_test, f"STATUS: {verdict}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    print(f"\n--- Inspection Result ---")
    print(f"Total Defect Area: {total_defect_area} px")
    print(f"Final Decision: {verdict}")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

align_and_diff('master.jpg', 'test_defect.jpg')