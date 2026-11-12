import pybullet as p
import pybullet_data
import time

def run_workcell():
    # 1. Start the physics simulation engine
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 2. Spawn the environment and the Robot Arm (Kuka IIWA)
    planeId = p.loadURDF("plane.urdf")
    robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    num_joints = p.getNumJoints(robotId)
    
    print(f"Successfully spawned Kuka Arm with {num_joints} joints.")

    # 3. Define the Inspection Waypoints (Joint angles for the arm)
    # Pose 1: Reaching down (simulating grabbing/cycling the test part)
    pose_cycle = [0, 0.5, 0, -1.5, 0, 1.0, 0]
    # Pose 2: Lifting up to present the part to a theoretical camera
    pose_inspect = [1.0, 0.5, 0, -1.0, 0, 1.5, 0]

    # 4. Run the Life-Cycle Testing Loop
    cycles = 5
    for i in range(cycles):
        print(f"\n--- Starting Cycle {i+1}/{cycles} ---")
        
        # Move to cycle/grab pose
        print("Moving to actuating position...")
        p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=pose_cycle)
        
        # Step physics to allow arm to move
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
            
        # Move to inspect pose
        print("Moving to inspection camera viewpoint...")
        p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=pose_inspect)
        
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
            
        print(f"Cycle {i+1} completed successfully.")
        time.sleep(1)

    print("All life-cycle tests completed. Shutting down workcell.")
    p.disconnect()

if __name__ == '__main__':
    run_workcell()