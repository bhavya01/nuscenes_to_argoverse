import argparse
from collections import defaultdict
import json

import numpy as np
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET

from argoverse.utils.manhattan_search import compute_point_cloud_bbox
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils


"""
Converts the nuScenes map into the Argoverse format.
The nuscenes map is a json file, and is converted to an xml for use with argoverse codebase.

"""

filename_to_id = {'boston-seaport':'BSP_10318', 'singapore-hollandvillage':'SHV_10320',
                  'singapore-onenorth':'SON_10322', 'singapore-queenstown':'SQT_10324'}
def populate_lane_dict(nusc_map, root, data):
    """
    Return a map that links a nuscenes 'way' token to all the nodes that are a part of that way.
    Also adds all these nodes to the xml map.
    """
    lane_dict = defaultdict(list)

    # Map x,y value of node to the new(argo) node id in the xml
    present_nodes = {}  # k:tuple, v: node_id
    global_id = 0  # New id for nodes in the xml

    # Loop over all lanes in the json to populate the lane to node dictionary
    for way in data['lane']:
        if way['token'] == '8f23d3ed-1089-4fcf-a178-a174abc69938':
            continue
        lane_record = nusc_map.get_lane(way['token'])  # get arcline associated with lane
        poses = arcline_path_utils.discretize_lane(lane_record,
                                                   resolution_meters=0.5)  # discretize the lane to given resolution
        for pose in poses:
            currNode = (pose[0], pose[1])
            if currNode not in present_nodes:
                node = ET.SubElement(root, 'node')
                node.set('id', str(global_id))
                present_nodes[currNode] = global_id
                global_id += 1
                node.set('x', str(pose[0]))
                node.set('y', str(pose[1]))
            lane_dict[way['token']].append(present_nodes[currNode])
    return lane_dict

def populate_polys(data):
    """
    Returns a dictionary poly_dict
    poly_dict maps a lane polygon in the original map to its exterior points.
    """
    # Map nodes in the original json to their x, y coordinates
    node_dict = {}  # k: node_token v: (x,y)

    # Loop over all nodes in the json to populate the node_dict
    for node in data['node']:
        node_dict[node['token']] = (node['x'], node['y'])

    # Map polygons in the original json to the x,y coordinates of their exterior nodes
    poly_dict = {}  # k: poly_token v: np.ndarray of shape (N,2)

    # Loop over all nodes in the json to populate the node_dict
    for poly in data['polygon']:
        poly_array = []
        for node_id in poly['exterior_node_tokens']:
            poly_array.append(list(node_dict[node_id]))
        poly_dict[poly['token']] = np.array(poly_array)
    return poly_dict

def create_lanes_xml(nusc_map, root, data, filename, argo_dir, lane_dict, poly_dict):
    """
    Fill up the xml map file with lane centelines.
    Also create the supporting files halluc_bbox_table.npy and tableidx_to_laneid_map.json
    """
    # Id to assign to lanes in the new xml map file. We arbitrarily start with 8000000.
    # We make up new lane_ids since the original one's are non numerical
    global_way_id = 8000000

    # Map that links new lane_id in the xml to its original token in the json.
    way_to_lane_id = {}  # k: global_way_id , v: data['lane']['token']

    # map lane segment IDs to their index in the table
    tableidx_to_laneid_map = {}
    # array that holds xmin,ymin,xmax,ymax for each coord
    halluc_bbox_table = []
    table_idx_counter = 0

    ## Iterate over the lanes to create the required xml and supporting files
    for way in data['lane']:
        if way['token'] == '8f23d3ed-1089-4fcf-a178-a174abc69938':
            continue
        node = ET.SubElement(root, 'way')

        if way['token'] not in way_to_lane_id:
            way_to_lane_id[way['token']] = global_way_id
            global_way_id += 1
        curr_id = way_to_lane_id[way['token']]

        node.set('lane_id', str(curr_id))
        traffic = ET.SubElement(node, 'tag')
        traffic.set('k', "has_traffic_control")
        traffic.set('v', "False")

        turn = ET.SubElement(node, 'tag')
        turn.set('k', "turn_direction")
        turn.set('v', "NONE")

        intersection = ET.SubElement(node, 'tag')
        intersection.set('k', "is_intersection")
        intersection.set('v', "False")

        ln = ET.SubElement(node, 'tag')
        ln.set('k', "l_neighbor_id")
        ln.set('v', "None")

        rn = ET.SubElement(node, 'tag')
        rn.set('k', "r_neighbor_id")
        rn.set('v', "None")

        for waypoint in lane_dict[way['token']]:
            nd = ET.SubElement(node, 'nd')
            nd.set('ref', str(waypoint))

        predecessors = nusc_map.get_incoming_lane_ids(way['token'])
        successors = nusc_map.get_outgoing_lane_ids(way['token'])

        for pred_id in predecessors:
            pre = ET.SubElement(node, 'tag')
            pre.set('k', "predecessor")
            if pred_id not in way_to_lane_id:
                way_to_lane_id[pred_id] = global_way_id
                global_way_id += 1
            int_pred_id = way_to_lane_id[pred_id]
            pre.set('v', str(int_pred_id))

        for succ_id in successors:
            succ = ET.SubElement(node, 'tag')
            succ.set('k', "successor")
            if succ_id not in way_to_lane_id:
                way_to_lane_id[succ_id] = global_way_id
                global_way_id += 1
            int_succ_id = way_to_lane_id[succ_id]
            succ.set('v', str(int_succ_id))

        lane_id = way_to_lane_id[way['token']]
        tableidx_to_laneid_map[table_idx_counter] = lane_id
        table_idx_counter += 1

        xmin, ymin, xmax, ymax = compute_point_cloud_bbox(poly_dict[way['polygon_token']])
        halluc_bbox_table += [(xmin, ymin, xmax, ymax)]

    halluc_bbox_table = np.array(halluc_bbox_table)
    halluc_bbox_dict = {
        "tableidx_to_laneid_map": tableidx_to_laneid_map,
        "halluc_bbox_table": halluc_bbox_table
    }
    np.save(f"{argo_dir}/{filename_to_id[filename]}_halluc_bbox_table.npy", halluc_bbox_table)
    with open(f"{argo_dir}/{filename_to_id[filename]}_tableidx_to_laneid_map.json", 'w') as outfile:
        json.dump(tableidx_to_laneid_map, outfile)

    tree = ET.ElementTree(root)
    with open(f"{argo_dir}/pruned_nuscenes_{filename_to_id[filename]}_vector_map.xml", "wb") as files:
        tree.write(files)

def convert_map(args):

    # Load json file. Loop over all the json files in directory
    for filename in ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']:
        print(filename)
        with open(f"/{args.nuscenes_dir}/maps/{filename}.json") as f:
            data = json.load(f)
        nusc_map = NuScenesMap(dataroot=f"/{args.nuscenes_dir}", map_name=filename)

        # Create new xml ETree
        root = ET.Element('NuScenesMap')

        # Map lane token of nuscenes to node_tokens that form the centerline
        lane_dict = populate_lane_dict(nusc_map, root, data)    # k: token, v: list of node_token

        poly_dict = populate_polys(data)

        create_lanes_xml(nusc_map, root, data, filename, args.argo_dir, lane_dict, poly_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuscenes-dir",
        default = "/coc/dataset/nuScenes-v1.0/nuScenes-map-expansion-v1.2",
        type=str,
        help="the path to the directory where the NuScenes map is stored",
    )
    parser.add_argument(
        "--argo-dir",
        default="output",
        type=str,
        help="the path to the directory where the converted data should be written",
    )
    args = parser.parse_args()
    convert_map(args)
