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

    # 6. Compute Structural Similarity (SSIM) to find defects
    score, diff_map = ssim(master, aligned_test, full=True)
    
    # Convert diff_map to standard 0-255 image format
    diff_map = (diff_map * 255).astype("uint8")
    
    # Threshold the diff map to highlight major differences (defects)
    _, defect_mask = cv2.threshold(diff_map, 150, 255, cv2.THRESH_BINARY_INV)

    # 7. Visualize the results
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(master, cmap='gray'); axes[0].set_title("Master")
    axes[1].imshow(test, cmap='gray'); axes[1].set_title("Raw Test (Shifted)")
    axes[2].imshow(aligned_test, cmap='gray'); axes[2].set_title("Aligned Test")
    axes[3].imshow(defect_mask, cmap='hot'); axes[3].set_title(f"Defect Heatmap\nSSIM: {score:.3f}")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

align_and_diff('master.jpg', 'test_defect.jpg')