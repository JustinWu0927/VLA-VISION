import cv2

def evaluate_lighting_robustness(master_path, test_path):
    master = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if master is None or test is None:
        print("Error: Could not load images.")
        return

    # Extract SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(master, None)
    kp2, des2 = sift.detectAndCompute(test, None)

    # Use BFMatcher with crossCheck=True (Two-sided symmetric matching)
    # It ensures that Feature A in img1 best matches Feature B in img2, 
    # AND Feature B best matches Feature A.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance (confidence)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate Confidence Metrics
    total_master_features = len(kp1)
    surviving_matches = len(matches)
    survival_rate = (surviving_matches / total_master_features) * 100 if total_master_features > 0 else 0

    print(f"--- Lighting Stress Test Results ---")
    print(f"Master Features Found: {total_master_features}")
    print(f"Robust Matches Surviving: {surviving_matches}")
    print(f"Feature Survival Rate: {survival_rate:.1f}%")
    
    if surviving_matches > 50:
        print("Status: EXCELLENT. System can confidently align these images.")
    elif surviving_matches > 15:
        print("Status: MARGINAL. System might align, but is prone to snapping errors.")
    else:
        print("Status: FAIL. Lighting is too degraded. Homography will break.")

    # Visualize the top 30 matches to see what the computer is grabbing onto
    img_matches = cv2.drawMatches(master, kp1, test, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow("Top 30 Surviving Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Run the function ---
# Test this against your shadow, dim, and glare photos.
evaluate_lighting_robustness('master.jpg', 'test_shadow.jpg')