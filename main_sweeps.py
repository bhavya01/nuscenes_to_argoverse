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
from main import get_calibration_info, round_to_micros, extract_pc
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

CITY_TO_ID = {
    "singapore-onenorth": "SON",
    "boston-seaport": "BSP",
    "singapore-queenstown": "SQT",
    "singapore-hollandvillage": "SHV",
}


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

        log_token = scene['log_token']
        nusc_log = nusc.get('log', log_token)
        nusc_city = nusc_log['location']
        with open(os.path.join(scene_path, f"city_info.json"), 'w') as f:
            json.dump({"city_name": CITY_TO_ID[nusc_city]}, f)

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
            nsweeps_lidar = 10
            nsweeps_cam = 6

            # Save ego pose to json file
            poses_path = os.path.join(scene_path, f"poses")
            if not os.path.exists(poses_path):
                os.makedirs(poses_path)

            # Copy nuscenes sensor data into argoverse format and get the pose of the vehicle in the city frame
            for sensor, sensor_token in sample['data'].items():    
                if sensor in SENSOR_NAMES:
                    argo_sensor = SENSOR_NAMES[sensor]
                    output_sensor_path = os.path.join(scene_path, argo_sensor)
                    if not os.path.exists(output_sensor_path):
                        os.makedirs(output_sensor_path)
                    sensor_data = nusc.get('sample_data', sensor_token)
                    file_path = os.path.join(NUSCENES_ROOT, sensor_data['filename'])
                    i = 0
                    if sensor == 'LIDAR_TOP' :
                        # nuscenes lidar data is stored as (x, y, z, intensity, ring index)
                        while i < nsweeps_lidar and sensor_token!= '' :
                            sensor_data = nusc.get('sample_data', sensor_token)
                            file_path = os.path.join(NUSCENES_ROOT, sensor_data['filename'])
                            timestamp = round_to_micros(sensor_data['timestamp'])
                            # Not always exactly 10
                            if (sensor_data["is_key_frame"] and i!=0) or sample_token== '':
                                break
                                #raise AssertionError("Less than 9 sweeps betweeen Lidar samples. Check sweep token no.", sensor_data['token'])
                            scan = np.fromfile(file_path, dtype=np.float32)
                            points = scan.reshape((-1, 5))

                            # Transform lidar points from point sensor frame to egovehicle frame
                            calibration = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
                            egovehicle_R_lidar = quat2rotmat(calibration['rotation'])
                            egovehicle_t_lidar = np.array(calibration['translation'])
                            egovehicle_SE3_lidar = SE3(rotation=egovehicle_R_lidar, translation=egovehicle_t_lidar)
                            points_egovehicle = egovehicle_SE3_lidar.transform_point_cloud(points[:, :3])

                            extract_pc(points_egovehicle, points, output_sensor_path, timestamp)

                            if not os.path.isfile(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json")) :
                                ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
                                ego_pose_dict = {"rotation": ego_pose["rotation"], "translation": ego_pose["translation"]}
                                with open(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json"), 'w') as f:
                                    json.dump(ego_pose_dict, f)

                            sensor_token = sensor_data['next']
                    else:
                        while i < nsweeps_cam and sensor_token != '':
                            sensor_data = nusc.get('sample_data', sensor_token)
                            file_path = os.path.join(NUSCENES_ROOT, sensor_data['filename'])
                            timestamp = round_to_micros(sensor_data['timestamp'])
                            # Not always exactly 6
                            if sensor_data["is_key_frame"] and i!=0:
                                #raise AssertionError("Less than 5 sweeps betweeen camera samples. Check sweep token no.", sensor_data['token'])
                                break
                            shutil.copy(file_path, os.path.join(output_sensor_path, f"{argo_sensor}_{timestamp}.jpg"))
                            sensor_token = sensor_data['next']

                            if not os.path.isfile(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json")) :
                                ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
                                ego_pose_dict = {"rotation": ego_pose["rotation"], "translation": ego_pose["translation"]}
                                with open(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json"), 'w') as f:
                                    json.dump(ego_pose_dict, f)


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
