import rclpy
from sensor_msgs.msg import Image as ROSImg, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
from rclpy.node import Node
import struct
import trimesh
import os
from datetime import datetime
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import copy

class DetectionNode(Node):

    # Base path for all STL files
    BASE_STL_PATH = "/home/ubuntu/ros2_ws/src/my_code/Models-2"
    # New path for saving pointclouds
    SAVE_PATH = "/home/ubuntu/ros2_ws/src/my_code/saved_pointclouds"
    
    # YOLO class names mapping
    CLASS_NAMES = [
        'X1-Y1-Z2', 'X1-Y2-Z1', 'X1-Y2-Z2', 'X1-Y2-Z2-CHAMFER', 
        'X1-Y2-Z2-TWINFILLET', 'X1-Y3-Z2', 'X1-Y3-Z2-FILLET', 
        'X1-Y4-Z1', 'X1-Y4-Z2', 'X2-Y2-Z2', 'X2-Y2-Z2-FILLET', 
        'undefined'
    ]

    def __init__(self):
        super().__init__('detection_node')
        
        # Initialize storage for detections and pointclouds
        self.current_detections = []
        self.detection_pointclouds = []
        self.latest_point_cloud = None
        self.stl_pointclouds = []
        self.processed_classes = set()

        # For ICP
        self.max_icp_iterations = 50
        self.icp_tolerance = 1e-6
        self.object_poses = []  # Will store the latest poses

        self.vis_publisher = self.create_publisher(
            ROSImg,
            '/object_pose_visualization',
            10
        )
        
        # Detection counter for each class (reset every new frame)
        self.detection_counters = {}
        
        # Create save directory if it doesn't exist
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        
        self.subscription_img = self.create_subscription(
            ROSImg,
            '/camera/image_raw/image',
            self.detect,
            10)
        self.subscription_pcl = self.create_subscription(
            PointCloud2,
            '/camera/image_raw/points',
            self.process_point_cloud,
            10)
            
        self.model = YOLO("/home/ubuntu/ros2_ws/src/my_code/localization/yolo_weights/best.pt")
        self.bridge = CvBridge()
        
        # Create session timestamp for grouping related pointclouds
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_pointcloud_to_ply(self, points, label, pc_type, instance_num=None):
        """
        Save pointcloud to PLY file with organized naming scheme
        
        Args:
            points: numpy array of points
            label: class name or 'scene' for full pointcloud
            pc_type: type of pointcloud ('raw', 'processed', 'stl_model')
            instance_num: instance number for multiple detections of same class
        """
        if len(points) == 0:
            self.get_logger().warn(f"Empty pointcloud, not saving {label}_{pc_type}")
            return
            
        # Ensure points are in numpy array format
        points = np.array(points)
        
        # Create the header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header"
        ]
        
        # Build filename
        if instance_num is not None:
            filename = f"{label}_i{instance_num}_{pc_type}"
        else:
            filename = f"{label}_{pc_type}"
            
        full_path = os.path.join(self.SAVE_PATH, f"{self.session_id}_{filename}.ply")
        
        # Save the file
        with open(full_path, 'w') as f:
            f.write('\n'.join(header) + '\n')
            np.savetxt(f, points, fmt='%.6f %.6f %.6f')
            
        self.get_logger().info(f"Saved pointcloud to {full_path}")

    # Load .stl files of the predicted labels, then convert from stl to pointcloud
    def load_stl_model(self, class_idx):
        try:
            # Skip if class was already processed or is 'undefined'
            if class_idx in self.processed_classes or class_idx >= len(self.CLASS_NAMES):
                return False
                
            class_name = self.CLASS_NAMES[class_idx]
            if class_name == 'undefined':
                return False
                
            # Construct the STL file path
            stl_path = os.path.join(
                self.BASE_STL_PATH,
                class_name,
                'mesh',
                f'{class_name}.stl'
            )
            
            # Check if file exists
            if not os.path.exists(stl_path):
                self.get_logger().warn(f"STL file not found: {stl_path}")
                return False
            
            # Load the mesh and convert to pointcloud
            mesh = trimesh.load(stl_path)
            points = mesh.sample(5000)  # Sample 5000 points from the surface

            # Compute centroid
            if len(points) > 0:
                centroid = np.mean(points, axis=0)
                self.get_logger().info(f"Centroid for {class_name}: {centroid}")
            else:
                centroid = np.array([0., 0., 0.])
                self.get_logger().warn(f"Empty pointcloud for {class_name}, using zero centroid")
            
            # Store pointcloud with its label
            self.stl_pointclouds.append({
                'label': class_name,
                'points': points,
                'class_idx': class_idx
            })
            
            # Mark this class as processed
            self.processed_classes.add(class_idx)
            
            self.get_logger().info(
                f"Successfully loaded STL model for {class_name}\n"
                f"Centroid: x={centroid[0]:.3f}"
            )
            return True
        
            
        except Exception as e:
            self.get_logger().error(f"Error loading STL file for class {class_idx}: {e}")
            return False

    def detect(self, ros2_img):
        try:
            # Reset detection counters for new frame
            self.detection_counters = {}
            
            cv_image = self.bridge.imgmsg_to_cv2(ros2_img, desired_encoding='passthrough')
            results = self.model.predict(source=cv_image, conf=0.25)
            
            self.current_detections = []
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.CLASS_NAMES[cls]

                    # Load and save STL model pointcloud (only once per class)
                    if self.load_stl_model(cls):
                        stl_points = self.stl_pointclouds[-1]['points']
                        self.save_pointcloud_to_ply(
                            stl_points,
                            class_name,
                            'stl_model'
                        )
                    
                    detection_info = {
                        'bbox': xyxy,
                        'confidence': conf,
                        'class': cls,
                        'class_name': class_name
                    }
                    self.current_detections.append(detection_info)
                    
                if self.latest_point_cloud is not None:
                    self.extract_pointcloud_portions()
                    
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")

    def process_point_cloud(self, msg):
        """Store the pointcloud and save it"""
        self.latest_point_cloud = msg
        
        # Extract and save full pointcloud
        points = []
        for row in range(msg.height):
            for col in range(msg.width):
                offset = row * msg.row_step + col * msg.point_step
                (x, y, z) = struct.unpack_from('fff', msg.data, offset)
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    points.append([x, y, z])
        
        # Save the full pointcloud
        self.save_pointcloud_to_ply(points, 'scene', 'full')
        
        # Process detections if we have any
        if self.current_detections:
            self.extract_pointcloud_portions()
    
    def preprocess_pointcloud(self, points, voxel_size=0.01):
        try:
            # Convert input to numpy array if it isn't already
            points_array = np.array(points)
            
            # Check if points array is empty
            if len(points_array) == 0:
                self.get_logger().warn("Empty pointcloud received in preprocessing")
                return np.array([])
                
            # Check if points array has the right shape
            if len(points_array.shape) != 2 or points_array.shape[1] != 3:
                self.get_logger().error(f"Invalid pointcloud shape: {points_array.shape}")
                return np.array([])
            
            # Remove outliers based on statistical analysis
            mean = np.mean(points_array, axis=0)
            std = np.std(points_array, axis=0)
            inliers_mask = np.all(np.abs(points_array - mean) <= 1.5 * std, axis=1)
            filtered_points = points_array[inliers_mask]
            
            # Check if we still have points after outlier removal
            if len(filtered_points) == 0:
                self.get_logger().warn("No points remained after outlier removal")
                return np.array([])
            
            # Remove points near the ground plane (shadow removal)
            z_threshold = np.percentile(filtered_points[:, 2], 10)
            above_ground_mask = filtered_points[:, 2] > z_threshold
            filtered_points = filtered_points[above_ground_mask]
            
            # Final check for empty array
            if len(filtered_points) == 0:
                self.get_logger().warn("No points remained after ground removal")
                return np.array([])
                
            return filtered_points
            
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud preprocessing: {str(e)}")
            return np.array([])

    def extract_pointcloud_portions(self):
        if not self.latest_point_cloud or not self.current_detections:
            return
            
        self.detection_pointclouds = []
        
        for detection in self.current_detections:
            class_name = detection['class_name']
            
            # Update counter for this class
            if class_name not in self.detection_counters:
                self.detection_counters[class_name] = 0
            instance_num = self.detection_counters[class_name]
            self.detection_counters[class_name] += 1
            
            bbox = detection['bbox']
            margin = 4
            
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(self.latest_point_cloud.width, x2 + margin)
            y2 = min(self.latest_point_cloud.height, y2 + margin)
            
            # Extract points for this detection
            points = []
            for row in range(y1, y2):
                for col in range(x1, x2):
                    offset = row * self.latest_point_cloud.row_step + col * self.latest_point_cloud.point_step
                    (x, y, z) = struct.unpack_from('fff', self.latest_point_cloud.data, offset)
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        points.append([x, y, z])
            
            if points:
                points_array = np.array(points)
                
                # Save raw portion
                self.save_pointcloud_to_ply(
                    points_array,
                    class_name,
                    'raw',
                    instance_num
                )
                
                # Process and save processed portion
                processed_points = self.preprocess_pointcloud(points_array)
                self.save_pointcloud_to_ply(
                    processed_points,
                    class_name,
                    'processed',
                    instance_num
                )
                
                # Store processed points for further use
                self.detection_pointclouds.append({
                    'points': processed_points,
                    'class_idx': detection['class'],
                    'bbox': detection['bbox']
                })

        # Run ICP
        if self.detection_pointclouds:
            self.process_poses()


    def visualize_depth_image_from_points(self, points, idx):
        image_height = 480
        image_width = 640
        depth_image = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for x, y, z in points:
            # Map the x, y, z values to image coordinates
            img_x = int(x * 100 + image_width // 2)
            img_y = int(y * 100 + image_height // 2)

            if np.isfinite(z) and 0 <= img_x < image_width and 0 <= img_y < image_height:
                depth_value = int((z + 2) * 40)
                depth_image[img_y, img_x] = min(max(depth_value, 0), 255)
        
        # Display the depth image
        cv2.imshow(f"Depth Image for Detection {idx}", depth_image)
        cv2.waitKey(1)

    # ICP code

    def estimate_initial_pose(self, source_points, target_points):
        """Estimate initial alignment based on principal components."""
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        source_covariance = np.cov(source_centered.T)
        target_covariance = np.cov(target_centered.T)
        
        source_eigvals, source_eigvecs = np.linalg.eigh(source_covariance)
        target_eigvals, target_eigvecs = np.linalg.eigh(target_covariance)
        
        R = target_eigvecs @ source_eigvecs.T
        
        if np.linalg.det(R) < 0:
            source_eigvecs[:, -1] *= -1
            R = target_eigvecs @ source_eigvecs.T
        
        t = target_centroid - (R @ source_centroid)
        
        return R, t

    def find_closest_points(self, source_points, target_points):
        """Find closest point correspondences using KD-tree."""
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        
        good_matches_mask = distances < np.percentile(distances, 90)
        
        return target_points[indices[good_matches_mask]], source_points[good_matches_mask]

    def calculate_transformation(self, source_points, target_points):
        """Calculate optimal rigid transformation between point sets."""
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = target_centroid - (R @ source_centroid)
        
        return R, t

    def apply_transformation(self, points, R, t):
        """Apply rigid transformation to points."""
        return (R @ points.T).T + t

    def calculate_error(self, source_points, target_points):
        """Calculate RMSE between point sets."""
        distances = np.linalg.norm(source_points - target_points, axis=1)
        return np.sqrt(np.mean(distances**2))

    def icp_align(self, source_points, target_points):
        """Main ICP algorithm implementation."""
        # Check for empty point clouds
        if len(source_points) < 3 or len(target_points) < 3:
            self.get_logger().warn("Not enough points for ICP alignment")
            return None, None, float('inf'), False
            
        try:
            R_current, t_current = self.estimate_initial_pose(source_points, target_points)
            
            transformed_source = copy.deepcopy(source_points)
            prev_error = float('inf')
            
            for iteration in range(self.max_icp_iterations):
                transformed_source = self.apply_transformation(source_points, R_current, t_current)
                
                matched_target, matched_source = self.find_closest_points(transformed_source, target_points)
                
                if len(matched_source) < 3:  # Need at least 3 points for rigid transformation
                    return None, None, float('inf'), False
                
                R_step, t_step = self.calculate_transformation(matched_source, matched_target)
                
                R_current = R_step @ R_current
                t_current = R_step @ t_current + t_step
                
                current_error = self.calculate_error(matched_source, matched_target)
                
                if abs(prev_error - current_error) < self.icp_tolerance:
                    return R_current, t_current, current_error, True
                    
                prev_error = current_error
                
            return R_current, t_current, prev_error, False
            
        except Exception as e:
            self.get_logger().error(f"Error in ICP alignment: {str(e)}")
            return None, None, float('inf'), False

    def extract_pose_parameters(self, R, t):
        """Convert rotation matrix and translation vector to interpretable parameters."""
        position = t.flatten()
        rotation = Rotation.from_matrix(R)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()
        
        return {
            'position': position,
            'euler_angles': euler_angles,
            'quaternion': quaternion
        }

    def process_poses(self):
        """Process all detected objects and estimate their poses."""

        self.get_logger().info(f"Starting pose processing. Found {len(self.detection_pointclouds)} detections")
        
        # Print debug information
        for i, detected in enumerate(self.detection_pointclouds):
            self.get_logger().info(f"Detection {i} keys: {detected.keys()}")
            self.get_logger().info(f"Number of points: {len(detected['points'])}")
        
        for i, stl_model in enumerate(self.stl_pointclouds):
            self.get_logger().info(f"STL model {i} keys: {stl_model.keys()}")
            self.get_logger().info(f"Class idx: {stl_model['class_idx']}")

        self.object_poses = []
        
        for detected in self.detection_pointclouds:
            
            stl_model = next((model for model in self.stl_pointclouds 
                            if model['class_idx'] == detected['class_idx']), None)
            
            if stl_model is None:
                self.get_logger().warn(f"No STL model found for class {detected['class_idx']}")
                continue
                
            # Align pointclouds
            R, t, error, converged = self.icp_align(
                stl_model['points'],
                detected['points']
            )
            
            if not converged or R is None or t is None:
                self.get_logger().warn(f"ICP failed for object {self.CLASS_NAMES[detected['class_idx']]}")
                continue
                
            # Extract pose parameters
            pose_params = self.extract_pose_parameters(R, t)
            
            self.object_poses.append({
                'class_idx': detected['class_idx'],
                'class_name': self.CLASS_NAMES[detected['class_idx']],
                'position': pose_params['position'],
                'euler_angles': pose_params['euler_angles'],
                'quaternion': pose_params['quaternion'],
                'alignment_error': error,
                'bbox': detected['bbox']
            })
            
            # Log pose information
            self.get_logger().info(
                f"\nPose estimated for {self.CLASS_NAMES[detected['class_idx']]}:\n"
                f"Position (x,y,z): {pose_params['position']}\n"
                f"Orientation (r,p,y): {pose_params['euler_angles']}\n"
                f"Alignment error: {error}"
            )
            
            # Visualize the results
            self.visualize_pose(detected['points'], stl_model['points'], R, t)

    def visualize_pose(self, scene_points, model_points, R, t):
        """Visualize the alignment results."""
        try:
            # Create a visualization image
            vis_img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Project 3D points to 2D for visualization
            # (This is a simple orthographic projection - adjust parameters as needed)
            scale = 100  # Scale factor for visualization
            offset = np.array([320, 240])  # Center of image
            
            # Draw original points (red)
            for point in scene_points:
                x, y = (point[:2] * scale + offset).astype(int)
                if 0 <= x < 640 and 0 <= y < 480:
                    cv2.circle(vis_img, (x, y), 1, (0, 0, 255), -1)
            
            # Draw transformed model points (green)
            transformed_points = self.apply_transformation(model_points, R, t)
            for point in transformed_points:
                x, y = (point[:2] * scale + offset).astype(int)
                if 0 <= x < 640 and 0 <= y < 480:
                    cv2.circle(vis_img, (x, y), 1, (0, 255, 0), -1)
            
            # Convert to ROS image and publish
            ros_img = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
            self.vis_publisher.publish(ros_img)
            
            # Also show in OpenCV window for debugging
            cv2.imshow('ICP Alignment Result', vis_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error in pose visualization: {str(e)}")

  

def main(args=None):
    rclpy.init(args=args)
    detector = DetectionNode()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("Shutting down DetectionNode...")
    finally:
        cv2.destroyAllWindows()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()