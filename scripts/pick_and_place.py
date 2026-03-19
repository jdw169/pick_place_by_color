#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import copy
import tf.transformations
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from darknet_ros_msgs.msg import BoundingBoxes
from kortex_driver.srv import *
from kortex_driver.msg import *

standby_pose = [0.320, 0.165, 0.323, 3.14, 0, 3.14]
pick_up_x = 0
pick_up_y = 0
pick_yaw = 0
drop_off_x = 0
drop_off_y = 0

class pick_cube_vision:

    def __init__(self):
        # Initialize C++ backend and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('pick_place_by_color', anonymous=True)

        # Define the namespace AND the exact path to the robot description
        robot_ns = "my_gen3_lite"
        robot_desc = "my_gen3_lite/robot_description"

        # Instantiate MoveIt objects, passing BOTH the namespace and the desc
        self.robot = moveit_commander.RobotCommander(robot_description=robot_desc, ns=robot_ns)
        self.scene = moveit_commander.PlanningSceneInterface(ns=robot_ns)
        
        self.move_group = moveit_commander.MoveGroupCommander("arm", robot_description=robot_desc, ns=robot_ns)
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper", robot_description=robot_desc, ns=robot_ns)
        # set maximum speed and acceleration 
        self.move_group.set_max_velocity_scaling_factor(0.8)
        self.move_group.set_max_acceleration_scaling_factor(0.8)
        # Give the scene a second to initialize before acting
        rospy.sleep(2) 
        """
        # clear faults
        self.robot_name = rospy.get_param('~robot_name', "my_gen3lite") 
        clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_full_name)
        self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

        rospy.loginfo("Clear faults")
        self.clear_faults()
        rospy.sleep(2.5)
        """
        # Move robot to home position
        rospy.loginfo("Moving arm to home position...")
        self.move_group.set_named_target("home")
        self.move_group.go(wait=True)
        # set move tolerance
        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.01)
        # Move to standby pose
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

        # Set up the TF2 Listener to translate coordinates
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Get the camera's lens properties (intrinsics)
        #self.camera_model = PinholeCameraModel()
        #cam_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        #self.camera_model.fromCameraInfo(cam_info)

        self.camera_model = PinholeCameraModel()
        dummy_info = CameraInfo()
        dummy_info.K = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
        dummy_info.P = [615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_model.fromCameraInfo(dummy_info)
        # flag to indicating the robot is busy with motion
        self.ispicking = False

        # Subscribe to Darknet bounding boxes
        self.bbox_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.vision_callback)


    def operate_gripper(self, state):
        # 1. Get the active motorized joint
        motorized_joint = self.gripper_group.get_active_joints()[0]
        
        # 2. Build a strict dictionary to force MoveIt's formatting
        # Gen3 Lite ranges from 0.0 (open) to approx 0.96 (tightly closed)
        target_dict = {}
        if state == "opened":
            target_dict[motorized_joint] = 0.80   
        elif state == "closed":
            target_dict[motorized_joint] = 0.20 

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


    def pick_and_place(self,pick_up_x,pick_up_y,drop_off_x,drop_off_y):
        rospy.loginfo("Starting Pick-and-Place sequence...")
        #standby_pose = self.move_group.get_current_pose().pose

        # 1. Ensure gripper is open before approaching
        self.operate_gripper("opened")

        # 2. Hover directly over the cube,
        target_pose = self.move_group.get_current_pose().pose
        target_pose.position.x = pick_up_x
        target_pose.position.y = pick_up_y
        target_pose.position.z = 0.015
        rospy.loginfo("Move clamp closer...")
        self.move_group.set_pose_target(target_pose)
        success = self.move_group.go(wait=True)
        if not success:
            rospy.logerr("Failed to move clamp closer! Stopping sequence.")
            return # Exit the function so it doesn't keep going


        # 4. Dip down to "grab" the part
        waypoints = []
        wpose = self.move_group.get_current_pose().pose
        #wpose.position.z -= 0.2
        waypoints.append(copy.deepcopy(wpose))
        (plan,fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01)
        if fraction < 0.9:
            rospy.logerr(f"only planned {fraction*100}% of the lift! Aborting")
            return
        rospy.loginfo("Moving to drop-off zone...")
        self.move_group.execute(plan, wait = True)
        
        """
        target_pose.position.z = 0.045 # Lower down toward the bed
        rospy.loginfo("Dipping to grasp height...")
        self.move_group.set_pose_target(target_pose)
        success = self.move_group.go(wait=True)
        if not success:
            rospy.logerr("Failed to dip down! Stopping sequence.")
            return # Exit the function so it doesn't keep going
        """
        # 5. Close the gripper
        
        self.operate_gripper("closed")

        # 6. Lift the part back up
        target_pose.position.z = 0.25
        rospy.loginfo("Lifting part...")
        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)

        # 7. Move to the intermediat waypoint (x=0 y=0.3)
        target_pose.position.x = 0.3
        target_pose.position.y = 0.0
        
        rospy.loginfo("Move to intermediat waypoint")
        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)
        
        # 8. Move to drop off zone
        target_pose.position.x = drop_off_x
        target_pose.position.y = drop_off_y
        rospy.loginfo("Move to drop off zone")
        self.move_group.set_pose_target(target_pose)
        self.move_group.go(wait=True)

        # 9. Open gripper to drop the part
        self.operate_gripper("opened")

        # 10. Return to the Standby Position
        
        rospy.loginfo("Returning to standby position...")
        self.move_group.set_named_target("home")
        #self.move_group.set_pose_target(standby_pose)
        self.move_group.go(wait=True)
        self.operate_gripper("closed")
        rospy.loginfo("Pick-and-Place complete! ...")

    def vision_callback(self, msg):
        # if the robot is in motion, return
        if self.ispicking == True:
            return
        
        for box in msg.bounding_boxes:
            if box.Class in ["kite","tvmonitor","cup"]:
                # 1. Find the center pixel of the bounding box
                self.ispicking = True
                center_u = (box.xmin + box.xmax) / 2.0
                center_v = (box.ymin + box.ymax) / 2.0
                depth_z = box.z / 1000 # Your depth value in meters
                
                # 2. Convert the 2D pixel into a 3D Ray, then multiply by Depth
                # This gives (X, Y, Z) in the camera's physical frame
                ray = self.camera_model.projectPixelTo3dRay((center_u, center_v))
                camera_x = ray[0] * depth_z
                camera_y = ray[1] * depth_z
                camera_z = depth_z
                
                # 3. Package this coordinate into a ROS PointStamped message
                target_point_camera = PointStamped()
                target_point_camera.header.frame_id = "camera_color_optical_frame"
                target_point_camera.header.stamp = rospy.Time.now()
                target_point_camera.point.x = camera_x
                target_point_camera.point.y = camera_y
                target_point_camera.point.z = camera_z
                
                # 4. Use TF2 to translate from the camera frame to the robot base frame
                try:
                    # Wait up to 1 second for the transform tree to be ready
                    target_point_base = self.tf_buffer.transform(target_point_camera, "my_gen3_lite/base_link", rospy.Duration(1.0))
                    
                    rospy.loginfo(f"Cube found at Robot X:{target_point_base.point.x:.2f}, Y:{target_point_base.point.y:.2f}")
                    
                    # 5. SEND TO MOVEIT!
                    # Now you can pass these exact X and Y coordinates to your pick_and_place function
                    # self.pick_and_place(target_point_base.point.x, target_point_base.point.y,0.15,-0.25)
                    
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logerr(f"TF Transform failed: {e}")
                self.ispicking  = False
                break


if __name__ == '__main__':
    try:
        pick_cube_vision()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass