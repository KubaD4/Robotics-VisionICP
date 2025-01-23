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
import time
from localization_interfaces.srv import BlockInfoAll

class DetectionNode(Node):
    """
    ROS2 node for object detection and pose estimation using YOLO and ICP.
    Processes camera images and point clouds to detect objects and estimate their 3D poses.
    """
    
    BASE_STL_PATH = "/home/ubuntu/ros2_ws/src/my_code/Models-2"
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
        self.latest_image = None
        self.stl_pointclouds = []
        self.processed_classes = set()
        self.object_poses = []

        # ICP parameters
        self.max_icp_iterations = 300
        self.icp_tolerance = 1e-7

        # Service
        self.srv = self.create_service(BlockInfoAll, 'localization_service', self.handle_all_block_info_request)

        # Publishers and subscribers setup
        self.vis_publisher = self.create_publisher(ROSImg, '/object_pose_visualization', 5)
        self.subscription_img = self.create_subscription(ROSImg, '/camera/image_raw/image', self.store_image, 5)
        self.subscription_pcl = self.create_subscription(PointCloud2, '/camera/image_raw/points', self.store_pointcloud, 5)
        
        self.model = YOLO("/home/ubuntu/ros2_ws/src/my_code/localization/yolo_weights/best.pt")
        self.bridge = CvBridge()

        self.get_logger().info("Detection node initialized and waiting for service requests...") # Idle --> waiting for a request 

    def store_image(self, msg):
        """Just store the latest image"""
        self.latest_image = msg

    def store_pointcloud(self, msg):
        """Just store the latest pointcloud"""
        self.latest_point_cloud = msg

    def handle_all_block_info_request(self, request, response):
        """Handle service request for block information"""
        try:
            self.get_logger().info("Received block info request - starting processing")
            
            # Check if we have both required messages
            step = 0
            step_limit = 10 # 10 seconds
            while step < step_limit:
                if self.latest_image is not None and self.latest_point_cloud is not None:
                    self.get_logger().warn(f"image and pointcloud ready after {step} seconds\n")
                    break  # Both data are ready, exit loop
                time.sleep(1)  # Wait for 1 second before checking again
                step += 1
                self.get_logger().warn(f"image or pointcloud still not ready, waited {step} seconds\n")
            # if after 10 seconds image or pointcloud still not ready, return
            if self.latest_image is None or self.latest_point_cloud is None:
                self.get_logger().error("No camera messages available")
                return response

            # Process the latest data
            self.detect(self.latest_image)
            
            # Wait for processing to complete
            max_attempts = 100  # 10 seconds
            attempt = 0
            
            while attempt < max_attempts:
                if self.object_poses:
                    # Calculate distances and create block info list
                    block_infos = []
                    for pose in self.object_poses:
                        position = pose['position']
                        distance = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
                        
                        block_infos.append({
                            'distance': distance,
                            'name': pose['class_name'],
                            'position': position,
                            'quaternion': pose['quaternion']
                        })

                    # Sort by distance from camera (descending)
                    block_infos.sort(key=lambda x: x['distance'], reverse=True)

                    # Fill response
                    response.block_names = [block['name'] for block in block_infos]
                    response.positions_x = [float(block['position'][0]) for block in block_infos]
                    response.positions_y = [float(block['position'][1]) for block in block_infos]
                    response.positions_z = [float(block['position'][2]) for block in block_infos]
                    response.orientations_x = [float(block['quaternion'][0]) for block in block_infos]
                    response.orientations_y = [float(block['quaternion'][1]) for block in block_infos]
                    response.orientations_z = [float(block['quaternion'][2]) for block in block_infos]
                    response.orientations_w = [float(block['quaternion'][3]) for block in block_infos]
                    
                    self.get_logger().info(f"Successfully processed {len(block_infos)} blocks\n\n")
                    
                    # Clear data for next request
                    self.object_poses = []
                    self.current_detections = []
                    self.detection_pointclouds = []
                    self.get_logger().info("Detection node waiting for service requests...") # Idle --> waiting for a request 
                    return response
                
                attempt += 1
                time.sleep(0.1)
            
            # If we get here, processing timed out
            self.get_logger().warn("Processing timed out - no blocks detected")
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error processing request: {str(e)}")
            return response

    def load_stl_model(self, class_idx):
        """Load and convert STL model to pointcloud for a given class index"""
        if class_idx in self.processed_classes or class_idx >= len(self.CLASS_NAMES):
            return False
                
        class_name = self.CLASS_NAMES[class_idx]
        if class_name == 'undefined':
            return False
                
        stl_path = os.path.join(self.BASE_STL_PATH, class_name, 'mesh', f'{class_name}.stl')
        
        if not os.path.exists(stl_path):
            self.get_logger().warn(f"STL file not found: {stl_path}")
            return False
        
        mesh = trimesh.load(stl_path)
        points = mesh.sample(5000)

        if len(points) > 0:
            self.stl_pointclouds.append({
                'label': class_name,
                'points': points,
                'class_idx': class_idx
            })
            self.processed_classes.add(class_idx)
            return True
        
        return False

    def detect(self, ros2_img):
        """Process incoming images for object detection using YOLO"""
        try:
            self.get_logger().warn("DETECT")
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

                    # Load STL model if not already loaded
                    self.load_stl_model(cls)
                    
                    self.current_detections.append({
                        'bbox': xyxy,
                        'confidence': conf,
                        'class': cls,
                        'class_name': class_name
                    })
                    
                if self.latest_point_cloud is not None:
                    self.extract_pointcloud_portions()
                    
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")

    def process_point_cloud(self, msg):
        """Store incoming point cloud data and process if detections exist"""
        self.get_logger().warn("PROCESS")
        self.latest_point_cloud = msg
        if self.current_detections:
            self.extract_pointcloud_portions()
    
    def preprocess_pointcloud(self, points, voxel_size=0.01):
        """Remove outliers and ground plane from pointcloud"""
        try:
            points_array = np.array(points)
            
            # Check if points array is empty or if points array has the right shape
            if len(points_array) == 0:
                return np.array([])
            if len(points_array.shape) != 2 or points_array.shape[1] != 3:
                return np.array([])
            
            # Statistical outlier removal
            mean = np.mean(points_array, axis=0)
            std = np.std(points_array, axis=0)
            inliers_mask = np.all(np.abs(points_array - mean) <= 1.5 * std, axis=1)
            filtered_points = points_array[inliers_mask]
            
            if len(filtered_points) == 0:
                return np.array([])
            
            # Ground plane removal
            z_threshold = np.percentile(filtered_points[:, 2], 10)
            above_ground_mask = filtered_points[:, 2] > z_threshold
            filtered_points = filtered_points[above_ground_mask]
            
            return filtered_points
            
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud preprocessing: {str(e)}")
            return np.array([])

    def extract_pointcloud_portions(self):
        """Extract point cloud segments for each detected object"""
        if not self.latest_point_cloud or not self.current_detections:
            return
            
        self.detection_pointclouds = []
        
        for detection in self.current_detections:
            bbox = detection['bbox']
            margin = 4
            
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2 = min(self.latest_point_cloud.width, x2 + margin)
            y2 = min(self.latest_point_cloud.height, y2 + margin)
            
            points = []
            for row in range(y1, y2):
                for col in range(x1, x2):
                    offset = row * self.latest_point_cloud.row_step + col * self.latest_point_cloud.point_step
                    (x, y, z) = struct.unpack_from('fff', self.latest_point_cloud.data, offset)
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        points.append([x, y, z])
            
            if points:
                points_array = np.array(points)
                processed_points = self.preprocess_pointcloud(points_array)
                
                self.detection_pointclouds.append({
                    'points': processed_points,
                    'class_idx': detection['class'],
                    'bbox': detection['bbox']
                })

        # Run ICP
        if self.detection_pointclouds:
            self.process_poses()

    # Start of the section to compute ICP
    def estimate_initial_pose(self, source_points, target_points):
        """Calculate initial alignment using principal component analysis"""
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
        """Find corresponding points using KD-tree"""
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        good_matches_mask = distances < np.percentile(distances, 90)
        return target_points[indices[good_matches_mask]], source_points[good_matches_mask]

    def calculate_transformation(self, source_points, target_points):
        """Calculate optimal rigid transformation between point sets"""
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
        """Apply rigid transformation to point set"""
        return (R @ points.T).T + t

    def calculate_error(self, source_points, target_points):
        """Calculate RMSE between point sets"""
        distances = np.linalg.norm(source_points - target_points, axis=1)
        return np.sqrt(np.mean(distances**2))

    def icp_align(self, source_points, target_points):
        """Perform Iterative Closest Point alignment"""
        if len(source_points) < 3 or len(target_points) < 3:
            self.get_logger().warn("Not enough points for ICP alignment")
            return None, None, float('inf'), False
            
        try:
            start_time = time.time()  # Start timing
            R_current, t_current = self.estimate_initial_pose(source_points, target_points)
            transformed_source = copy.deepcopy(source_points)
            prev_error = float('inf')
            
            for iteration in range(self.max_icp_iterations):
                transformed_source = self.apply_transformation(source_points, R_current, t_current)
                matched_target, matched_source = self.find_closest_points(transformed_source, target_points)
                
                if len(matched_source) < 3:
                    return None, None, float('inf'), False
                
                R_step, t_step = self.calculate_transformation(matched_source, matched_target)
                R_current = R_step @ R_current
                t_current = R_step @ t_current + t_step
                
                current_error = self.calculate_error(matched_source, matched_target)
                
                if abs(prev_error - current_error) < self.icp_tolerance:
                    end_time = time.time()  # End timing
                    self.get_logger().info(f"ICP alignment completed. Time taken: {end_time - start_time:.4f} seconds") 
                    return R_current, t_current, current_error, True
                    
                prev_error = current_error
                
            end_time = time.time()  # End timing
            self.get_logger().info(f"ICP alignment completed. Time taken: {end_time - start_time:.4f} seconds")    
            return R_current, t_current, prev_error, False
            
        except Exception as e:
            self.get_logger().error(f"Error in ICP alignment: {str(e)}")
            return None, None, float('inf'), False

    def extract_pose_parameters(self, R, t):
        """Convert rotation matrix and translation to pose parameters"""
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
        """Process detected objects and estimate their poses using ICP"""
        self.object_poses = []
        
        for detected in self.detection_pointclouds:
            stl_model = next((model for model in self.stl_pointclouds 
                            if model['class_idx'] == detected['class_idx']), None)
            
            if stl_model is None:
                self.get_logger().warn(f"No STL model found for class {detected['class_idx']}")
                continue
                
            R, t, error, converged = self.icp_align(stl_model['points'], detected['points'])
            
            if not converged or R is None or t is None:
                self.get_logger().warn(f"ICP failed for object {self.CLASS_NAMES[detected['class_idx']]}")
                continue
                
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
            
            self.get_logger().info(
                f"\nPose estimated for {self.CLASS_NAMES[detected['class_idx']]}:\n"
                f"Position (x,y,z): {pose_params['position']}\n"
                f"Orientation (r,p,y): {pose_params['euler_angles']}\n"
                f"Alignment error: {error}"
            )

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