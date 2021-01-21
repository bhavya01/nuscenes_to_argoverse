# nuScenes to Argoverse Converter
This repository provides a set of tools to convert all nuScenes raw data and map data to the Argoverse format. There are 3 main scripts:
- nuScenes `samples` (2 hz) converter: `main.py`
- nuScenes `sweeps` (20 hz) and `samples` (2 hz) converter: `main_sweeps.py`
- map converter: `map_conversion.py`

<p align="left">
  <img src="https://user-images.githubusercontent.com/15787503/96957491-c79e7f00-14c8-11eb-8eaf-32c2b01f4124.gif" height="215">
  <img src="https://user-images.githubusercontent.com/15787503/96957715-63c88600-14c9-11eb-9476-8469c4b2fe17.gif" height="215">
</p>
<p align="left">
  <img src="https://user-images.githubusercontent.com/15787503/96957958-11d43000-14ca-11eb-9b25-e58798d5f318.gif" height="215">
  <img src="https://user-images.githubusercontent.com/15787503/96958977-f585c280-14cc-11eb-8f38-1853e30bd1de.gif" height="215">
</p>

## Dependencies

[nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit/blob/bccad8f0ee19afd963f41ae36133ae05516a7ed3/docs/installation.md)
[argoverse-api](https://github.com/argoai/argoverse-api#installation)

## Data Format Overview

NuScenes dataset stores the raw data in a relational database as described [here](https://github.com/nutonomy/nuscenes-devkit/blob/bccad8f0ee19afd963f41ae36133ae05516a7ed3/docs/schema_nuscenes.md).
Lidar data is provided in the sensor coordinate system and each point has 5 dimensions. The x,y,z coordinates, intensity and ring index. The intensity measures the reflectivity of the objects and ring index is the index of the laser ranging from 0 to 31.
Annotations are provided in the global city coordinate frame.
