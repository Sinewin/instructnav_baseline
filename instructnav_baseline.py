"""
InstructNav Baseline for ROS2-based Unity Simulator
Integrates the full InstructNav framework (Mapper + LLM planning) with ROS2 communication.
Compatible with Unity simulator through SimulatorCommand messages.
"""

import rclpy
import numpy as np
import cv2
import ast
import threading
import open3d as o3d
from typing import Dict, Any, List
import quaternion

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from simulator_messages.msg import SimulatorCommand

from ros2.src.vln_connector.vln_connector.vln_connector import VLNConnector
from ros2.src.vln_connector.vln_connector.events import event_manager

# Import your existing InstructNav components from baselines directory
import sys
import os

# Add path to import your InstructNav modules
from baselines.mapping_utils.geometry import *
from baselines.mapping_utils.projection import *
from baselines.mapping_utils.path_planning import *
from baselines.mapping_utils.transform import unity_rotation, unity_translation
from baselines.constants import INTEREST_OBJECTS

# Import LLM utils if available
try:
    from baselines.llm_utils.nav_prompt import CHAINON_PROMPT, GPT4V_PROMPT
    from baselines.llm_utils.gpt_request import gpt_response, gptv_response
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM utils import failed: {e}")
    LLM_AVAILABLE = False

# Import Mapper
try:
    from baselines.mapper import Mapper
    MAPPER_AVAILABLE = True
except ImportError as e:
    print(f"Mapper import failed: {e}")
    MAPPER_AVAILABLE = False


class InstructNavMapperROS:
    """
    InstructNav Mapper wrapper for ROS2 environment.
    Handles mapping, LLM planning, and coordinate transformations.
    """
    
    def __init__(self, camera_intrinsic):
        if not MAPPER_AVAILABLE:
            raise ImportError("Mapper not available. Check baselines/mapper.py")
        
        # Initialize Mapper with Unity coordinate system
        self.mapper = Mapper(
            camera_intrinsic=camera_intrinsic,
            pcd_resolution=0.05,
            grid_resolution=0.1,
            grid_size=5,
            floor_height=-0.8,
            ceiling_height=0.8,
            translation_func=unity_translation,  # Unity to Mapper
            rotation_func=unity_rotation,        # Unity to Mapper
            rotate_axis=[0, 1, 0],
            device='CPU:0',
            enable_object_detection=True
        )
        
        # State variables
        self.initialized = False
        self.observed_objects = []
        self.trajectory_summary = ""
        self.current_instruction = ""
        
        # Panoramic data collection
        self.temporary_images = []  # Store images for panoramic view
        self.temporary_pcd = []     # Store point clouds for each direction
        
        # Trajectory recording (for debugging)
        self.rgb_trajectory = []
        self.depth_trajectory = []
        
        # Planning state
        self.plan_step = 0
        self.max_steps = 500
        self.lock = threading.RLock()
        
        # Current pose
        self.current_position = None
        self.current_rotation = None
        self.current_yaw = 0.0
        
        print("InstructNavMapperROS initialized")
        print(f"  - LLM available: {LLM_AVAILABLE}")
    
    def reset(self, position, rotation):
        """Reset mapper to initial state"""
        with self.lock:
            self.mapper.reset(position, rotation)
            self.initialized = True
            self.trajectory_summary = ""
            self.plan_step = 0
            self.temporary_images = []
            self.temporary_pcd = []
            self.rgb_trajectory = []
            self.depth_trajectory = []
            self.observed_objects = []
            print("Mapper reset")
    
    def update(self, rgb, depth, position, rotation):
        """Update mapper with new observation"""
        with self.lock:
            # Initialize on first update
            if not self.initialized:
                self.reset(position, rotation)
            
            # Store current pose
            self.current_position = position
            self.current_rotation = rotation
            
            # Extract yaw from rotation matrix (for debugging)
            rot_matrix = unity_rotation(rotation)
            self.current_yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
            
            # Convert BGR to RGB for Mapper
            rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Preprocess depth
            depth = np.nan_to_num(depth, nan=5.0)
            depth = np.clip(depth, 0.1, 20.0)

            # Update mapper
            self.mapper.update(rgb_rgb, depth, position, rotation)
            
            # Record trajectory (for debugging)
            self.rgb_trajectory.append(rgb_rgb.copy())
            self.depth_trajectory.append((depth / 5.0 * 255.0).astype(np.uint8))
            
            # Update observed objects
            self.observed_objects = self.mapper.get_appeared_objects()
            
            # Collect panoramic data (store up to 12 directions)
            if len(self.temporary_images) < 12:
                self.temporary_images.append(rgb_rgb.copy())
                try:
                    view_pcd = self.mapper.current_pcd
                    self.temporary_pcd.append(view_pcd)
                except:
                    pass
    
    def translate_objnav(self, instruction: str) -> str:
        """Translate object navigation instruction (from objnav_agent.py)"""
        instruction_lower = instruction.lower()
        
        if 'plant' in instruction_lower:
            return "Find the potted_plant."
        elif 'tv' in instruction_lower or 'monitor' in instruction_lower:
            return "Find the television_set."
        elif 'chair' in instruction_lower:
            return "Find the chair."
        elif 'table' in instruction_lower:
            return "Find the table."
        elif 'sofa' in instruction_lower or 'couch' in instruction_lower:
            return "Find the sofa."
        elif 'door' in instruction_lower:
            return "Find the door."
        elif 'bed' in instruction_lower:
            return "Find the bed."
        else:
            # Extract object name from instruction
            words = instruction_lower.split()
            for word in words:
                if word in ['find', 'go', 'to', 'the', 'a', 'an']:
                    continue
                return f"Find the {word}."
            return instruction
    
    def query_chainon(self) -> Dict[str, Any]:
        """Query Chainon LLM for planning decision"""
        if not LLM_AVAILABLE:
            return {'Action': 'Explore', 'Landmark': 'unknown', 'Flag': False}
        
        semantic_clue = {'observed_objects': self.observed_objects}
        query_content = f"<Navigation Instruction>:{self.current_instruction}, <Previous Plan>:{self.trajectory_summary}, <Semantic Clue>:{semantic_clue}"
        
        for attempt in range(3):
            try:
                raw_answer = gpt_response(query_content, CHAINON_PROMPT)
                print(f"GPT-4 Response: {raw_answer[:200]}...")
                
                # Parse JSON response
                answer = raw_answer.replace(" ", "")
                start_idx = answer.find("{")
                end_idx = answer.find("}") + 1
                
                if start_idx != -1 and end_idx != 0:
                    answer_json = answer[start_idx:end_idx]
                    answer_dict = ast.literal_eval(answer_json)
                    
                    if all(k in answer_dict for k in ['Action', 'Landmark', 'Flag']):
                        # Update trajectory summary
                        if not self.trajectory_summary:
                            self.trajectory_summary = f"{answer_dict['Action']}-{answer_dict['Landmark']}"
                        else:
                            self.trajectory_summary += f"-{answer_dict['Action']}-{answer_dict['Landmark']}"
                        
                        print(f"Parsed: Action={answer_dict['Action']}, Landmark={answer_dict['Landmark']}, Flag={answer_dict['Flag']}")
                        return answer_dict
            except Exception as e:
                print(f"Chainon query error (attempt {attempt+1}): {e}")
                continue
        
        # Default fallback
        return {'Action': 'Explore', 'Landmark': 'unknown', 'Flag': False}
    
    def query_gpt4v(self) -> int:
        """Query GPT-4V for panoramic direction selection"""
        if not LLM_AVAILABLE or not self.temporary_images:
            return 0
        
        # Create panoramic image
        panoramic = self._create_panoramic_image(self.temporary_images)
        text_content = f"<Navigation Instruction>:{self.current_instruction}"
        
        for attempt in range(3):
            try:
                raw_answer = gptv_response(text_content, panoramic, GPT4V_PROMPT)
                print(f"GPT-4V Response: {raw_answer[:100]}...")
                
                # Parse direction from response
                if "Direction" in raw_answer:
                    for i in range(12):
                        if f"Direction {i}" in raw_answer or f"方向{i}" in raw_answer:
                            return i
            except Exception as e:
                print(f"GPT-4V query error (attempt {attempt+1}): {e}")
                continue
        
        return np.random.randint(0, min(12, len(self.temporary_images)))
    
    def _create_panoramic_image(self, images: List[np.ndarray]) -> np.ndarray:
        """Create panoramic image from multiple views"""
        if not images:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            height, width = images[0].shape[:2]
        except:
            height, width = 480, 640
        
        # Create 3x4 grid of images
        rows = 3
        cols = 4
        panoramic = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
        
        for i, img in enumerate(images[:rows*cols]):
            row = i // cols
            col = i % cols
            panoramic[row*height:(row+1)*height, col*width:(col+1)*width] = img
        
        return panoramic
    
    def plan_navigation(self, instruction: str):
        """
        Execute full InstructNav planning pipeline.
        Returns target point in mapper coordinates.
        """
        with self.lock:
            # Set current instruction
            self.current_instruction = instruction
            
            # Step 1: Query Chainon LLM
            print("Step 1: Querying Chainon LLM...")
            chainon_result = self.query_chainon()
            
            # Step 2: Query GPT-4V for direction (if we have panoramic data)
            print("Step 2: Querying GPT-4V for direction...")
            gpt4v_direction = 0
            if self.temporary_images:
                gpt4v_direction = self.query_gpt4v()
            
            # Step 3: Get GPT-4V selected point cloud
            print("Step 3: Getting GPT-4V selected point cloud...")
            gpt4v_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device)
            if 0 <= gpt4v_direction < len(self.temporary_pcd):
                gpt4v_pcd = self.temporary_pcd[gpt4v_direction]
            
            # Step 4: Calculate affordance map
            try:
                print("Step 4: Calculating affordance map...")
                affordance, _ = self.mapper.get_objnav_affordance_map(
                    chainon_result['Action'],
                    chainon_result['Landmark'],
                    gpt4v_pcd,
                    chainon_result['Flag']
                )
                
                # Step 5: Select target point
                print("Step 5: Selecting target point...")
                if affordance.max() > 0:
                    target_idx = np.argmax(affordance)
                    navigable_points = self.mapper.navigable_pcd.point.positions.cpu().numpy()
                    target_point = navigable_points[target_idx]
                    
                    print(f"✅ Selected target point: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}]")
                    return target_point
                    
            except Exception as e:
                print(f"Affordance calculation error: {e}")
            
            # Fallback: explore or return current position
            print("⚠️ Using fallback exploration")
            if hasattr(self.mapper, 'navigable_pcd') and not self.mapper.navigable_pcd.is_empty():
                navigable_points = self.mapper.navigable_pcd.point.positions.cpu().numpy()
                if len(navigable_points) > 0:
                    # Select a point 1 meter ahead
                    target_point = self.mapper.current_position.copy()
                    target_point[1] += 1.0  # Move 1 meter forward in mapper coordinates
                    return target_point
            
            # Last resort: stay in place
            return self.mapper.current_position.copy()
    
    def get_current_pose(self):
        """Get current pose in mapper coordinates"""
        with self.lock:
            return self.mapper.current_position.copy(), self.current_yaw


class InstructNavBaseline(VLNConnector):
    """
    ROS2-based InstructNav Baseline.
    Inherits from VLNConnector for ROS2 communication with Unity simulator.
    """
    
    def __init__(self):
        super().__init__()  # Initialize VLNConnector (ROS2 node)
        
        # Camera intrinsic (adjust based on your Unity camera)
        self.camera_intrinsic = np.array([
            [415.6922, 0, 320],
            [0, 415.6922, 240],
            [0, 0, 1]
        ])
        
        # Initialize InstructNav Mapper
        try:
            self.instruct_nav = InstructNavMapperROS(self.camera_intrinsic)
            print("✅ InstructNav Mapper initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Mapper: {e}")
            self.instruct_nav = None
        
        # Planning state
        self.current_instruction = None
        self.is_planning = False
        self.plan_lock = threading.Lock()
        
        # Inference thread (like agent_baseline.py)
        self._inference_thread = None
        self._stop_event = threading.Event()
        
        # Register event handlers
        event_manager.register("task_received", self.handle_task_received)
        
        self.get_logger().info("🎯 InstructNavBaseline ROS2 Node started")
        self.get_logger().info(f"  - Mapper available: {MAPPER_AVAILABLE}")
        self.get_logger().info(f"  - LLM available: {LLM_AVAILABLE}")
    
    def handle_task_received(self, task: str):
        """Handle incoming task from ROS2 topic"""
        if task and task.strip():
            self.current_instruction = task.strip()
            self.get_logger().info(f"📝 Task received: {self.current_instruction}")
            
            # Reset panoramic collection for new task
            if self.instruct_nav:
                self.instruct_nav.temporary_images = []
                self.instruct_nav.temporary_pcd = []
    
    def control_once_async(self):
        """Main control loop (called from main loop)"""
        # Skip if already planning or no instruction
        if self.is_planning or not self.current_instruction or not self.instruct_nav:
            return
        
        # Check if inference thread is running
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return
        
        # Get observation data
        obs = self.get_observation()
        if obs is None:
            self.get_logger().warning("Observation not ready", throttle_duration_sec=2.0)
            return
        
        # Start non-blocking inference thread
        self._inference_thread = threading.Thread(
            target=self._run_inference,
            kwargs={
                "rgb": obs["rgb"].copy() if obs["rgb"] is not None else None,
                "depth": obs["depth"].copy() if obs["depth"] is not None else None,
                "base_pose": obs["base_pose"]
            }
        )
        self._inference_thread.start()
    
    def _run_inference(self, rgb, depth, base_pose):
        """Run InstructNav inference in separate thread"""
        self.is_planning = True
        
        try:
            if rgb is None or depth is None or base_pose is None:
                self.get_logger().warning("Inference missing data")
                return
            
            # Extract position and rotation from base_pose
            # base_pose is (x, y, yaw) from robot_odom_callback
            x, y, yaw = base_pose
            
            # Convert to Unity format (position + quaternion)
            # Assuming ground plane (z=0) and converting yaw to quaternion
            position = (x, 0.0, y)  # Unity: (x, y, z) where y is up
            # Convert yaw to quaternion (rotation around Y axis in Unity)
            qw = np.cos(yaw / 2)
            qx = 0.0
            qy = np.sin(yaw / 2)  # Rotation around Y axis
            qz = 0.0
            rotation = (qx, qy, qz, qw)
            
            # Update Mapper with current observation
            self.instruct_nav.update(rgb, depth, position, rotation)
            
            # Run InstructNav planning
            target_point = self.instruct_nav.plan_navigation(self.current_instruction)
            
            # Convert target point to ROS coordinates for Unity
            ros_x, ros_y, ros_yaw_deg = self._mapper_to_ros(target_point)
            
            # Create and publish SimulatorCommand
            cmd = self._create_simulator_command(ros_x, ros_y, ros_yaw_deg)
            self.publish_simulator_command(cmd)
            
            self.get_logger().info(
                f"🚗 Published move command: "
                f"forward={ros_x:.2f}m, left={ros_y:.2f}m, rotate={ros_yaw_deg:.1f}°"
            )
            
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            self.is_planning = False
    
    def _mapper_to_ros(self, target_point):
        """
        Convert target point from mapper coordinates to ROS coordinates.
        
        Mapper coordinates (after unity_translation):
          X = right (positive to right)
          Y = forward (positive forward) 
          Z = up
        
        ROS coordinates for Unity move command:
          x = forward displacement (meters)
          y = left displacement (meters) - positive y means move left
          yaw = rotation angle (degrees) - positive = counterclockwise/left
        """
        # Get current pose from mapper
        current_pos, current_yaw = self.instruct_nav.get_current_pose()
        
        # Calculate displacement in mapper coordinates
        dx_mapper = target_point[0] - current_pos[0]  # right displacement
        dy_mapper = target_point[1] - current_pos[1]  # forward displacement
        
        # Convert to ROS coordinates:
        ros_x = dy_mapper      # Mapper forward -> ROS forward (x)
        ros_y = -dx_mapper     # Mapper right -> ROS left (y, negative)
        
        # Calculate yaw (direction to target)
        if abs(dx_mapper) < 0.01 and abs(dy_mapper) < 0.01:
            # No significant movement, don't rotate
            ros_yaw = 0.0
        else:
            # Calculate direction in mapper coordinates
            target_yaw_mapper = np.arctan2(dx_mapper, dy_mapper)
            # Convert to ROS: positive yaw = left turn
            ros_yaw = -target_yaw_mapper
        
        # Convert to degrees
        ros_yaw_deg = np.degrees(ros_yaw)
        
        # Limit movement to reasonable values
        distance = np.sqrt(ros_x**2 + ros_y**2)
        if distance > 2.0:
            scale = 2.0 / distance
            ros_x *= scale
            ros_y *= scale
            self.get_logger().warning(f"Limited movement from {distance:.2f}m to 2.0m")
        
        return ros_x, ros_y, ros_yaw_deg
    
    def _create_simulator_command(self, ros_x, ros_y, ros_yaw_deg):
        """Create SimulatorCommand message for Unity"""
        cmd = SimulatorCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "instructnav"
        cmd.method = "move"
        cmd.method_params = f"{ros_x:.3f},{ros_y:.3f},{ros_yaw_deg:.3f}"
        return cmd
    
    def destroy_node(self):
        """Cleanup on node destruction"""
        self._stop_event.set()
        
        # Wait for inference thread to finish
        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    """Main function to run the ROS2 node"""
    rclpy.init(args=args)
    node = InstructNavBaseline()
    
    try:
        # Main loop similar to agent_baseline.py
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)  # Process ROS2 messages
            node.control_once_async()               # Run non-blocking inference
            
            if node._stop_event.is_set():
                node.get_logger().info("🛑 Stopping baseline")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if node._inference_thread is not None:
            node.get_logger().info("Waiting for inference thread...")
            node._inference_thread.join(timeout=2.0)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()