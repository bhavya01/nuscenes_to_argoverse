import argparse
import json
import os
import shutil
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from pyquaternion import Quaternion

from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from nuscenes.nuscenes import NuScenes

"""
Converts the nuScenes dataset into the Argoverse format.

In the nuScenes dataset, samples contain annotated data sampled at a frequency
of 2Hz, whereas sweeps contains all the unannotated data. The capture frequency
for camera is 12Hz and for lidar it 20Hz. Lidar data is provided in lidar sensor
coordinate system, annotations are provided in the global city coordinate frame,
and images are undistorted.

See paper:
https://arxiv.org/pdf/1903.11027.pdf
"""

# nuscenes sensor on the left, corresponding argoverse sensor on right
SENSOR_NAMES = {
    'LIDAR_TOP': 'lidar',
    'CAM_FRONT': 'ring_front_center',
    'CAM_FRONT_LEFT': 'ring_front_left',
    'CAM_FRONT_RIGHT': 'ring_front_right',
    'CAM_BACK_LEFT': 'ring_side_left',
    'CAM_BACK_RIGHT': 'ring_side_right',
}

def get_argo_label(label: str) -> str:
    """Map the nuscenes labels to argoverse labels"""
    if 'human' in label:
        return 'PEDESTRIAN'
    if 'vehicle' in label:
        if 'bicycle' in label:
            return 'BICYCLE'
        if 'motorcycle' in label:
            return 'MOTORCYCLE'
        if 'emergency' in label:
            return 'EMERGENCY_VEHICLE'
        if 'truck' in label:
            return 'LARGE_VEHICLE'
        if 'bus' in label:
            return 'BUS'
        if 'trailer' in label:
            return 'TRAILER'
        return 'VEHICLE'
    if 'movable_object' in label:
        return 'ON_ROAD_OBSTACLE'
    if 'animal' in label:
        return 'ANIMAL'
    return 'UNKNOWN'

def is_camera_sensor(sensor_name: str) -> bool:
    return sensor_name[0:3] == "CAM"

def get_calibration_info(nusc: Type[NuScenes], scene: Dict[Any, Any]) -> Dict[Any, Any]:
    """Output calibration info for all the sensors for the scene"""
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    result = {}
    cam_data = []
    for nuscenes_sensor, argoverse_sensor in SENSOR_NAMES.items():
        sensor_data = nusc.get('sample_data', sample['data'][nuscenes_sensor])
        calibration = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
        transformation_dict = {}
        transformation_dict['rotation'] = {'coefficients': calibration['rotation']}
        transformation_dict['translation'] = calibration['translation']
        if is_camera_sensor(nuscenes_sensor):
            camera_info = {}
            camera_info['key'] = "image_raw_" + argoverse_sensor
            value = {}
            value["focal_length_x_px_"] = calibration['camera_intrinsic'][0][0]
            value["focal_length_y_px_"] = calibration['camera_intrinsic'][1][1]
            value["focal_center_x_px_"] = calibration['camera_intrinsic'][0][2]
            value["focal_center_y_px_"] = calibration['camera_intrinsic'][1][2]
            value["skew_"] = calibration['camera_intrinsic'][0][1]
            # Nuscenes does not provide distortion coefficients.
            value["distortion_coefficients_"] = [0, 0, 0]
            value['vehicle_SE3_camera_'] = transformation_dict
            camera_info['value'] = value
            cam_data.append(camera_info)
        else:
            # Nuscenes has one lidar sensor.
            result["vehicle_SE3_down_lidar_"] = transformation_dict
            result["vehicle_SE3_up_lidar_"] = transformation_dict
    result['camera_data_'] = cam_data
    return result

def round_to_micros(t_nanos: int, base: int = 1000) -> int:
    """
    Round nanosecond timestamp to nearest microsecond timestamp
    """
    return base * round(t_nanos / base)

def main(args: argparse.Namespace) -> None:
    OUTPUT_ROOT = args.argo_dir
    NUSCENES_ROOT = args.nuscenes_dir
    NUSCENES_VERSION = args.nuscenes_version

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=True)
    for scene in nusc.scene:
        scene_token = scene['token']
        sample_token = scene['first_sample_token']
        scene_path = os.path.join(OUTPUT_ROOT, scene_token)

        if not os.path.exists(scene_path):
            os.makedirs(scene_path)

        with open(os.path.join(scene_path, f"city_info.json"), 'w') as f:
            json.dump({"city_name": 'PIT'}, f)

        # Calibration info for all the sensors
        calibration_info = get_calibration_info(nusc, scene)
        calib_path = os.path.join(scene_path, f"vehicle_calibration_info.json")
        with open(calib_path, 'w') as f:
            json.dump(calibration_info, f)

        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            timestamp = round_to_micros(sample['timestamp'])
            tracked_labels = []

            # city_SE3_vehicle pose
            ego_pose = None

            # Copy nuscenes sensor data into argoverse format and get the pose of the vehicle in the city frame
            for sensor, sensor_token in sample['data'].items():    
                if sensor in SENSOR_NAMES:
                    argo_sensor = SENSOR_NAMES[sensor]
                    output_sensor_path = os.path.join(scene_path, argo_sensor)
                    if not os.path.exists(output_sensor_path):
                        os.makedirs(output_sensor_path)
                    sensor_data = nusc.get('sample_data', sensor_token)
                    file_path = os.path.join(NUSCENES_ROOT, sensor_data['filename'])
                    if sensor == 'LIDAR_TOP':
                        # nuscenes lidar data is stored as (x, y, z, intensity, ring index)
                        scan = np.fromfile(file_path, dtype=np.float32)
                        points = scan.reshape((-1, 5))

                        # Transform lidar points from point sensor frame to egovehicle frame
                        calibration = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
                        egovehicle_R_lidar = quat2rotmat(calibration['rotation'])
                        egovehicle_t_lidar = np.array(calibration['translation'])
                        egovehicle_SE3_lidar = SE3(rotation=egovehicle_R_lidar, translation=egovehicle_t_lidar)
                        points_egovehicle = egovehicle_SE3_lidar.transform_point_cloud(points[:, :3])

                        data = {"x": points_egovehicle[:, 0], "y": points_egovehicle[:, 1], "z": points_egovehicle[:, 2], "intensity": points[:,3]}
                        cloud = PyntCloud(pd.DataFrame(data))
                        cloud_fpath = os.path.join(output_sensor_path, f"PC_{timestamp}.ply")
                        cloud.to_file(cloud_fpath)
                    else: 
                        shutil.copy(file_path, os.path.join(output_sensor_path, f"{argo_sensor}_{timestamp}.jpg"))

                    if ego_pose is None:
                        ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])

            # Save ego pose to json file
            poses_path = os.path.join(scene_path, f"poses")
            if not os.path.exists(poses_path):
                os.makedirs(poses_path)

            ego_pose_dict = {"rotation": ego_pose["rotation"], "translation": ego_pose["translation"]}
            with open(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json"), 'w') as f:
                json.dump(ego_pose_dict, f)

            # Object annotations
            labels_path = os.path.join(scene_path, f"per_sweep_annotations_amodal")
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            
            for ann_token in sample['anns']:
                annotation = nusc.get('sample_annotation', ann_token)
                city_SE3_object = SE3(quat2rotmat(annotation['rotation']), np.array(annotation['translation']))
                city_SE3_egovehicle = SE3(quat2rotmat(ego_pose['rotation']), np.array(ego_pose['translation']))
                egovehicle_SE3_city = city_SE3_egovehicle.inverse()
                egovehicle_SE3_object = egovehicle_SE3_city.right_multiply_with_se3(city_SE3_object)

                x, y, z = egovehicle_SE3_object.translation
                qw, qx, qy, qz = Quaternion(matrix=egovehicle_SE3_object.rotation)
                width, length, height = annotation['size']
                label_class = annotation['category_name']

                tracked_labels.append({
                    "center": {"x": x, "y": y, "z": z},
                    "rotation": {"x": qx , "y": qy, "z": qz, "w": qw},
                    "length": length,
                    "width": width,
                    "height": height,
                    "track_label_uuid": annotation['instance_token'],
                    "timestamp": timestamp,
                    "label_class": get_argo_label(label_class)
                })

            json_fpath = os.path.join(labels_path, f"tracked_object_labels_{timestamp}.json")
            with open(json_fpath, 'w') as f:
                json.dump(tracked_labels, f)

            sample_token = sample['next']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuscenes-dir",
        default="nuscenes",
        type=str,
        help="the path to the directory where the NuScenes data is stored",
    )
    parser.add_argument(
        "--nuscenes-version",
        default="v1.0-mini",
        type=str,
        help="the version of the NuScenes data to convert",
    )
    parser.add_argument(
        "--argo-dir",
        default="nuscenes_to_argoverse/output",
        type=str,
        help="the path to the directory where the converted data should be written",
    )
    args = parser.parse_args()
    main(args)
