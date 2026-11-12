import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random

# --- AI CONFIGURATION ---
# 1. Rebuild the model architecture to match what we trained
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
# Load the weights you just trained
model.load_state_dict(torch.load('defect_model.pth'))
model.eval() # Set to evaluation (testing) mode

# 2. Match the exact image transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# PyTorch ImageFolder sorts classes alphabetically. 'fail' = 0, 'pass' = 1
CLASS_NAMES = ['FAIL (Defect Detected)', 'PASS (Normal)']

def simulate_camera_capture():
    """Simulates taking a photo by grabbing a random image from your dataset"""
    # Pick randomly between pass or fail folder to simulate manufacturing variance
    category = random.choice(['pass', 'fail'])
    folder_path = os.path.join('dataset', category)
    
    # Grab a random image from that folder
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    chosen_image = random.choice(images)
    full_path = os.path.join(folder_path, chosen_image)
    
    # Run the image through your trained neural network
    img = Image.open(full_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    return CLASS_NAMES[predicted.item()], full_path

# --- ROBOTICS CONFIGURATION ---
def run_integrated_workcell():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    
    pose_cycle = [0, 0.5, 0, -1.5, 0, 1.0, 0]
    pose_inspect = [1.0, 0.5, 0, -1.0, 0, 1.5, 0]

    cycles = 3
    print("\n[SYSTEM] Starting Unified Test-and-Inspect Workcell...")
    
    for i in range(cycles):
        print(f"\n================ CYCLE {i+1} ================")
        
        # 1. Life-Cycle Testing Phase
        print("[ARM] Moving to actuating position (Stress Testing Part)...")
        p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=pose_cycle)
        for _ in range(100): p.stepSimulation(); time.sleep(1./240.)
            
        # 2. Multi-View Inspection Phase
        print("[ARM] Moving to inspection camera viewpoint...")
        p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=pose_inspect)
        for _ in range(100): p.stepSimulation(); time.sleep(1./240.)
            
        print("[CAMERA] Taking high-res snapshot of part...")
        time.sleep(0.5) # Simulate camera exposure time
        
        # 3. Vision-Language-Action (VLA) Inference Phase
        verdict, img_source = simulate_camera_capture()
        print(f"[AI VISION] Analyzing frame: {img_source}")
        print(f"[AI DECISION] Part Status: {verdict}")
        
        if "FAIL" in verdict:
            print("[SYSTEM] Anomaly flagged! Logging defect to database.")
        
        time.sleep(1)

    p.disconnect()

if __name__ == '__main__':
    run_integrated_workcell()