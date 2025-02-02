# Vision and Localization System

## Overview
This module provides object detection and pose estimation for Lego blocks in a robotic workspace, using YOLO detection and ICP-based pose estimation. Part of the Fundamentals of Robotics course at the University of Trento.

## Features
- Real-time object detection using YOLO
- Point cloud processing and registration
- 6-DoF pose estimation using ICP
- ROS2 service integration
- Support for multiple block types

## System Requirements
- Docker installed and running
- ROS2 Humble
- Python 3.8+

## Setup and Installation

### 1. Start the Docker Environment
```bash
# Start the ROS2 container
bash scripts/ros2.sh

# Access via noVNC at http://localhost:6081
```

### 2. Configure the Environment
```bash
# Navigate to workspace
cd /home/ubuntu/ros2_ws

# Source ROS2
source install/setup.bash

# Launch simulation
ros2 launch ros2_ur5_interface sim.launch.py
```

### 3. Configure Block Visualization
- Add RobotModel from the Add menu
- Select Topic as the description source
- Choose the correct topic for block models

## Running the Vision System

### 1. Launch the Detection Node
```bash
# Start the detection node
ros2 run localization detector
```

### 2. Emulate request Block Information
```bash
ros2 service call /localization_service localization_interfaces/srv/BlockInfoAll "{}"
```

The service returns:
- Block classifications
- 3D positions (x, y, z)
- Orientations (quaternions)
- Error metrics

## Configuration

### Supported Block Types
```python
CLASS_NAMES = [
    'X1-Y1-Z2', 'X1-Y2-Z1', 'X1-Y2-Z2', 'X1-Y2-Z2-CHAMFER',
    'X1-Y2-Z2-TWINFILLET', 'X1-Y3-Z2', 'X1-Y3-Z2-FILLET',
    'X1-Y4-Z1', 'X1-Y4-Z2', 'X2-Y2-Z2', 'X2-Y2-Z2-FILLET'
]
```

## Troubleshooting

### Common Issues
1. **No blocks detected**
   - Check camera topics
   - Verify YOLO model path
   - Check lighting conditions

2. **Poor pose estimation**
   - Verify point cloud quality
   - Check STL model availability

3. **Service timeout**
   - Increase timeout duration
   - Check system resources


