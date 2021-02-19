impor argparse
import json
import math
import multiprocessing
import os
import shutil
import sys
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from pyquaternion import Quaternion

from argoverse.utils.json_utils import save_json_dict
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from main import get_calibration_info, round_to_micros, write_ply
from nuscenes.nuscenes import NuScenes

"""
Converts the nuScenes unannotated sweeps into Argoverse format.

In the nuScenes dataset, samples contain annotated data sampled at a frequency
of 2Hz, whereas sweeps contains all the unannotated data. The capture frequency
for camera is 12Hz and for LiDar it 20Hz. We also convert samples to unannotated 
Argoverse format.

"""

# nuscenes sensor on the left, corresponding argoverse sensor on right
SENSOR_NAMES = {
    "LIDAR_TOP": "lidar",
    "CAM_FRONT": "ring_front_center",
    "CAM_FRONT_LEFT": "ring_front_left",
    "CAM_FRONT_RIGHT": "ring_front_right",
    "CAM_BACK_LEFT": "ring_side_left",
    "CAM_BACK_RIGHT": "ring_side_right",
}

# 3-letter abbreviation of nuScene city names
CITY_TO_ID = {
    "singapore-onenorth": "SON",
    "boston-seaport": "BSP",
    "singapore-queenstown": "SQT",
    "singapore-hollandvillage": "SHV",
}


def main(nusc: NuScenes,args: argparse.Namespace, start_index: int, end_index: int) -> None:
    """
    Convert sweeps and samples into (unannotated) Argoverse format. Overview of algorithm:

    1) Iterate over all scenes in the NuScenes dataset. For each scene, obtain first sample in the scene.
    2) Get the sample_data corresponding to each of the channels from the sample, and convert it to argo format.
    3) While the sample_data is not corresponding to a key_frame, get the next sample_data, and repeat step 2.
    4) Go to the next sample while we are in the same scene.
    """
    OUTPUT_ROOT = args.argo_dir
    NUSCENES_ROOT = args.nuscenes_dir
    NUSCENES_VERSION = args.nuscenes_version

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    tot_scenes = len(nusc.scene)
    for scene in nusc.scene[start_index:min(end_index, tot_scenes)]:
        scene_token = scene["token"]
        sample_token = scene["first_sample_token"]
        scene_path = os.path.join(OUTPUT_ROOT, scene_token)

        if not os.path.exists(scene_path):
            os.makedirs(scene_path)

        log_token = scene["log_token"]
        nusc_log = nusc.get("log", log_token)
        nusc_city = nusc_log["location"]
        save_json_dict(os.path.join(scene_path, f"city_info.json"), {"city_name": CITY_TO_ID[nusc_city]})

        # Calibration info for all the sensors
        calibration_info = get_calibration_info(nusc, scene)
        calib_path = os.path.join(scene_path, f"vehicle_calibration_info.json")
        save_json_dict(calib_path, calibration_info)

        while sample_token != "":
            sample = nusc.get("sample", sample_token)
            timestamp = round_to_micros(sample["timestamp"])
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
            for sensor, sensor_token in sample["data"].items():
                if sensor in SENSOR_NAMES:
                    argo_sensor = SENSOR_NAMES[sensor]
                    output_sensor_path = os.path.join(scene_path, argo_sensor)
                    if not os.path.exists(output_sensor_path):
                        os.makedirs(output_sensor_path)
                    sensor_data = nusc.get("sample_data", sensor_token)
                    file_path = os.path.join(NUSCENES_ROOT, sensor_data["filename"])
                    i = 0
                    if sensor == "LIDAR_TOP":
                        # nuscenes lidar data is stored as (x, y, z, intensity, ring index)
                        while i < nsweeps_lidar and sensor_token != "":
                            sensor_data = nusc.get("sample_data", sensor_token)
                            file_path = os.path.join(NUSCENES_ROOT, sensor_data["filename"])
                            timestamp = round_to_micros(sensor_data["timestamp"])
                            # Not always exactly 10
                            if (sensor_data["is_key_frame"] and i != 0) or sample_token == "":
                                break
                            scan = np.fromfile(file_path, dtype=np.float32)
                            points = scan.reshape((-1, 5))

                            # Transform lidar points from point sensor frame to egovehicle frame
                            calibration = nusc.get(
                                "calibrated_sensor",
                                sensor_data["calibrated_sensor_token"],
                            )
                            egovehicle_R_lidar = quat2rotmat(calibration["rotation"])
                            egovehicle_t_lidar = np.array(calibration["translation"])
                            egovehicle_SE3_lidar = SE3(
                                rotation=egovehicle_R_lidar,
                                translation=egovehicle_t_lidar,
                            )
                            points_egovehicle = egovehicle_SE3_lidar.transform_point_cloud(points[:, :3])

                            write_ply(points_egovehicle, points, output_sensor_path, timestamp)

                            if not os.path.isfile(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json")):
                                ego_pose = nusc.get("ego_pose", sensor_data["ego_pose_token"])
                                ego_pose_dict = {
                                    "rotation": ego_pose["rotation"],
                                    "translation": ego_pose["translation"],
                                }

                                save_json_dict(
                                    os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json"), ego_pose_dict
                                )

                            sensor_token = sensor_data["next"]
                    else:
                        while i < nsweeps_cam and sensor_token != "":
                            sensor_data = nusc.get("sample_data", sensor_token)
                            file_path = os.path.join(NUSCENES_ROOT, sensor_data["filename"])
                            timestamp = round_to_micros(sensor_data["timestamp"])
                            # Not always exactly 6
                            if sensor_data["is_key_frame"] and i != 0:
                                break
                            shutil.copy(
                                file_path,
                                os.path.join(output_sensor_path, f"{argo_sensor}_{timestamp}.jpg"),
                            )
                            sensor_token = sensor_data["next"]

                            if not os.path.isfile(os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json")):
                                ego_pose = nusc.get("ego_pose", sensor_data["ego_pose_token"])
                                ego_pose_dict = {
                                    "rotation": ego_pose["rotation"],
                                    "translation": ego_pose["translation"],
                                }
                                save_json_dict(
                                    os.path.join(poses_path, f"city_SE3_egovehicle_{timestamp}.json"), ego_pose_dict
                                )

            sample_token = sample["next"]


if __name__ == "__main__":
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

    jobs = []

    NUSCENES_ROOT = args.nuscenes_dir
    NUSCENES_VERSION = args.nuscenes_version
    num_processes = 30
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=True)

    total_scenes =   len(nusc.scene)
    chunk_size = math.ceil(total_scenes / num_processes)
    print(f"Will divide {total_scenes} items between {num_processes} processes")
    for i in range(num_processes):
        start_index = chunk_size * i
        end_index = start_index + chunk_size
        p = multiprocessing.Process(
            target=main,
            args=(
                nusc,
                args,
                start_index,
                end_index,
            ),
        )
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    print("Finished")

