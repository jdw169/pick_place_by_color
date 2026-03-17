#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import copy
import tf.transformations

#standby_joint_pose = [0.663,-0.440,1.413,-1.571,-1.290,0.663]
standby_pose = [0.320,0.165,0.323,3.14,0,3.14]
pick_up_x = 0
pick_up_y = 0
pick_yaw = 0
drop_off_x = 0
drop_off_y = 0

class pick_cube:

    def __init__(self):
        # 1. Initialize C++ backend and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('pick_place_by_color', anonymous=True)

        # 2. Define the namespace AND the exact path to the robot description
        robot_ns = "my_gen3_lite"
        robot_desc = "my_gen3_lite/robot_description"

        # 3. Instantiate MoveIt objects, passing BOTH the namespace and the desc
        self.robot = moveit_commander.RobotCommander(robot_description=robot_desc, ns=robot_ns)
        self.scene = moveit_commander.PlanningSceneInterface(ns=robot_ns)
        
        self.move_group = moveit_commander.MoveGroupCommander("arm", robot_description=robot_desc, ns=robot_ns)
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper", robot_description=robot_desc, ns=robot_ns)
        # set maximum speed and acceleration 
        self.move_group.set_max_velocity_scaling_factor(0.8)
        self.move_group.set_max_acceleration_scaling_factor(0.8)
        # Give the scene a second to initialize before acting
        rospy.sleep(2) 
        
        # 4. Move to home position
        rospy.loginfo("Moving arm to home position...")
        self.move_group.set_named_target("home")
        self.move_group.go(wait=True)
        
        # 5. Move to standby pose
        #self.move_group.set_joint_value_target(standby_joint_pose)

        rospy.loginfo("Moving arm to standby position...Tilt end effector completely upright....")
        self.move_group.set_pose_target(standby_pose)
        success = self.move_group.go(wait=True)
        if not success:
            rospy.logerr("Failed to tilt clamp upright! Stopping sequence.")
            return # Exit the function so it doesn't keep going
        current_pose = self.move_group.get_current_pose().pose
        quaternion = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        rospy.loginfo(f"Current RPY (in radians): Roll={roll:.3f}, Pitch={pitch:.3f}, Yaw={yaw:.3f}")    
        self.move_group.stop()

        # 6. Start the pick and place sequence
        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.01)
        pick_up_x = 0.35
        pick_up_y = 0.1
        pick_yaw = 0
        drop_off_x = -0.2
        drop_off_y = 0.15
        self.pick_and_place(pick_up_x,pick_up_y,pick_yaw,drop_off_x,drop_off_y)

    def operate_gripper(self, state):
        # 1. Get the active motorized joint
        motorized_joint = self.gripper_group.get_active_joints()[0]
        
        # 2. Build a strict dictionary to force MoveIt's formatting
        # Gen3 Lite ranges from 0.0 (open) to approx 0.96 (tightly closed)
        target_dict = {}
        if state == "opened":
            target_dict[motorized_joint] = 0.80   
        elif state == "closed":
            target_dict[motorized_joint] = 0.08  
            
        rospy.loginfo(f"Commanding {motorized_joint} to: {target_dict[motorized_joint]}")
        
        # 3. Execute using the dictionary target
        self.gripper_group.set_joint_value_target(target_dict)
        success = self.gripper_group.go(wait=True)
        
        # 4. Stop any residual joint commands
        self.gripper_group.stop() 
        
        if success:
            rospy.loginfo("Gripper command sent to physics engine successfully.")
        else:
            rospy.logwarn("MoveIt failed to execute gripper command.")
            
        rospy.sleep(1) # Give the physics engine a second to settle

    def turn_end_effector(self,yaw):
        current_pose = self.move_group.get_current_pose().pose
        quaternion = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        (roll, pitch, current_yaw) = tf.transformations.euler_from_quaternion(quaternion)
        target_pose_list = [current_pose.position.x,
                            current_pose.position.y,
                            current_pose.position.z,
                            roll, pitch, yaw]
        self.move_group.set_pose_target(target_pose_list)
        success = self.move_group.go(wait=True)
        if not success:
            rospy.logerr("Failed to turn end effctor! Stopping sequence.")
            return

    def pick_and_place(self,pick_up_x,pick_up_y,pick_yaw,drop_off_x,drop_off_y):
        rospy.loginfo("Starting Pick-and-Place sequence...")
        #standby_pose = self.move_group.get_current_pose().pose

        # 1. Ensure gripper is open before approaching
        self.operate_gripper("opened")

        # 2. Hover directly over the cube,
        target_pose = self.move_group.get_current_pose().pose
        target_pose.position.x = pick_up_x
        target_pose.position.y = pick_up_y
        rospy.loginfo("Move clamp closer...")
        self.move_group.set_pose_target(target_pose)
        success = self.move_group.go(wait=True)
        if not success:
            rospy.logerr("Failed to move clamp closer! Stopping sequence.")
            return # Exit the function so it doesn't keep going

        # 3. turn end effector to align with the part

        self.turn_end_effector(pick_yaw)

        # 4. Dip down to "grab" the part
        
        target_pose.position.z = 0.045 # Lower down toward the bed
        rospy.loginfo("Dipping to grasp height...")
        self.move_group.set_pose_target(target_pose)
        success = self.move_group.go(wait=True)
        if not success:
            rospy.logerr("Failed to dip down! Stopping sequence.")
            return # Exit the function so it doesn't keep going
        # 5. Close the gripper
        self.operate_gripper("closed")

        # 6. Lift the part back up
        target_pose.position.z = 0.25
        rospy.loginfo("Lifting part...")
        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)

        # 7. Move to the intermediat waypoint (x=0 y=0.3)
        target_pose.position.y = 0.3
        target_pose.position.x = 0.0
        rospy.loginfo("Move to intermediat waypoint")
        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)
        
        # 8. Move to drop off zone
        target_pose.position.x = drop_off_x
        target_pose.position.y = drop_off_y
        rospy.loginfo("Move to drop off zone")
        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)
        """
        waypoints = []
        wpose = self.move_group.get_current_pose().pose
        wpose.position.x = -0.25
        waypoints.append(copy.deepcopy(wpose))
        (plan,fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01)
        if fraction < 0.9:
            rospy.logerr(f"only planned {fraction*100}% of the lift! Aborting")
            return
        rospy.loginfo("Moving to drop-off zone...")
        self.move_group.execute(plan, wait = True)
        """
        # 9. Open gripper to drop the part
        self.operate_gripper("opened")

        # 10. Return to the Standby Position
        rospy.loginfo("Returning to standby position...")
        self.move_group.set_pose_target(standby_pose)
        self.move_group.go(wait=True)
        self.operate_gripper("closed")
        rospy.loginfo("Pick-and-Place complete! ...")

if __name__ == '__main__':
    try:
        pick_cube()
        #rospy.spin()
    except rospy.ROSInterruptException:
        pass
