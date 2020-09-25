import json
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import shutil

OUTPUT_ROOT = 'nuscenes_to_argoverse/output'
NUSCENES_ROOT = 'nuscenes'

SENSOR_NAMES = {
    'LIDAR_TOP': 'lidar',
    'CAM_FRONT': 'ring_front_center',
    'CAM_FRONT_LEFT': 'ring_front_left',
    'CAM_FRONT_RIGHT': 'ring_front_right',
    'CAM_BACK_LEFT': 'ring_side_left',
    'CAM_BACK_RIGHT': 'ring_side_right',
}

def is_camera_sensor(sensor):
    return sensor[0:3] == "CAM"

def get_calibration_info(nusc, scene):
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
            value["skew"] = calibration['camera_intrinsic'][0][1]
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

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_ROOT, verbose=True)
    for scene in nusc.scene:
        scene_token = scene['token']
        sample_token = scene['first_sample_token']
        scene_path = os.path.join(OUTPUT_ROOT, scene_token)

        if not os.path.exists(scene_path):
            os.makedirs(scene_path)

        # Calibration info for all the sensors
        calibration_info = get_calibration_info(nusc, scene)
        calib_path = os.path.join(scene_path, f"vehicle_calibration_info.json")
        with open(calib_path, 'w') as f:
            json.dump(calibration_info, f)

        tracked_labels = []

        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            ego_pose = None
            # TODO: Convert Lidar data in argoverse format
            for sensor, sensor_token in sample['data'].items():    
                if sensor in SENSOR_NAMES:
                    argo_sensor = SENSOR_NAMES[sensor]
                    output_sensor_path = os.path.join(scene_path, argo_sensor)
                    if not os.path.exists(output_sensor_path):
                        os.makedirs(output_sensor_path)
                    sensor_data = nusc.get('sample_data', sensor_token)
                    file_path = sensor_data['filename']
                    if ego_pose is None:
                        ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
                    shutil.copy(os.path.join(NUSCENES_ROOT, file_path), output_sensor_path)

            # Object annotations
            labels_path = os.path.join(scene_path, f"per_sweep_annotations_amodal")
            if not os.path.exists(labels_path):
                        os.makedirs(labels_path)
            
            for ann_token in sample['anns']:
                annotation = nusc.get('sample_annotation', ann_token)
                box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))

                # Convert the box into ego_vehicle coordinates
                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)

                x, y, z = box.center
                width, length, height = box.wlh
                qw, qx, qy, qz = box.orientation.elements
                # TODO: Map nuscenes label to argoverse labels
                label_class = annotation['category_name']

                tracked_labels.append({
                    "center": {"x": x, "y": y, "z": z},
                    "rotation": {"x": qx , "y": qy, "z": qz, "w": qw},
                    "length": length,
                    "width": width,
                    "height": height,
                    "track_label_uuid": annotation['instance_token'],
                    "timestamp": sample['timestamp'],
                    "label_class": label_class
                })
            json_fpath = os.path.join(labels_path, f"tracked_object_labels_{sample['timestamp']}.json")
            with open(json_fpath, 'w') as f:
                json.dump(tracked_labels, f)


            sample_token = sample['next']

if __name__ == '__main__':
    main()