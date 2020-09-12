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
    'LIDAR_TOP': 'lidar'
    'CAM_FRONT': 'ring_front_center',
    'CAM_FRONT_LEFT': 'ring_front_left',
    'CAM_FRONT_RIGHT': 'ring_front_right',
    'CAM_BACK_LEFT': 'ring_side_left',
    'CAM_BACK_RIGHT': 'ring_side_right',
}

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

        tracked_labels = []

        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            ego_pose = None
            # TODO: Convert Lidar data in argoverse format
            # TODO: Check images are undistorted and rectified.
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