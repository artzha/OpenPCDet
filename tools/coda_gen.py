import os
import pdb
import json
import shutil
import argparse

import numpy as np
import open3d as o3d

CLASS_REMAP = {
    "Scooter":              "Scooter",
    "Bike":                 "Bike",
    "Car":                  "Vehicle",
    "Motorcycle":           "Motorcycle",
    "Golf Cart":            "Vehicle",
    "Truck":                "Vehicle",
    "Person":               "Person",
    # Static Classes     
    "Tree":                 "Tree",
    "Traffic Sign":         "Sign",
    "Canopy":               "Canopy",
    "Traffic Lights":       "Traffic Lights",
    "Bike Rack":            "Bike Rack",
    "Bollard":              "Barrier",
    "Construction Barrier": "Barrier",
    "Parking Kiosk":        "Dispenser",
    "Mailbox":              "Dispenser",
    "Fire Hydrant":         "Fire Hydrant",
    # Static Class Mixed
    "Freestanding Plant":   "Plant",
    "Pole":                 "Pole",
    "Informational Sign":   "Sign",
    "Door":                 "Barrier",
    "Fence":                "Barrier",
    "Railing":              "Barrier",
    "Cone":                 "Cone",
    "Chair":                "Chair",
    "Bench":                "Bench",
    "Table":                "Table",
    "Trash Can":            "Trash Can",
    "Newspaper Dispenser":  "Dispenser",
    # Static Classes Indoor
    "Room Label":           "Sign",
    "Stanchion":            "Barrier",
    "Sanitizer Dispenser":  "Dispenser",
    "Condiment Dispenser":  "Dispenser",
    "Vending Machine":      "Dispenser",
    "Emergency Aid Kit":    "Dispenser",
    "Fire Extinguisher":    "Dispenser",
    "Computer":             "Screen",
    "Television":           "Screen",
    "Other":                "Other"
}

parser = argparse.ArgumentParser()
parser.add_argument('--channels', default=128,
                    help="number of vertical lidar channels to use")

def generate_labels(in_root, out_root, trajectories, use_custom=False):
    for traj in trajectories:
        print("Generating labels for trajectory %d"%traj)
        meta_path = os.path.join(in_root, "metadata", "%s.json"%traj)
        assert os.path.isfile(meta_path), '%s does not exist' % meta_path

        meta_json = json.load(open(meta_path, "r"))
        split_keys = ["train", "val", "test"]

        for split in split_keys:
            split_dir = os.path.join(out_root, split, "labels")
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

            split_paths = meta_json["ObjectTracking"][split]
            for label_subpath in split_paths:
                label_path = os.path.join(in_root, label_subpath)

                label_json = json.load(open(label_path, "r"))
                label_list = label_json["3dbbox"]

                label_file = label_path.split("/")[-1]

                
                #Create output label file
                if use_custom:
                    cust_file_dir = os.path.join(out_root, "labels")
                    frame = int(label_file.split("_")[-1].split(".")[0])
                    cust_file   = "%06d.txt" % frame
                    cust_path   = os.path.join(cust_file_dir, cust_file)
                else:
                    cust_file  = label_file.replace(".json", ".txt")
                    cust_path  = os.path.join(split_dir, cust_file)
                
                cust_txt    = open(cust_path, "w+")
                for label in label_list:
                    #Write to output
                    x, y, z     = label["cX"], label['cY'], label['cZ'] - 1.2 # Lower labels by 1.2meters in z
                    dx, dy, dz  = label["l"], label["w"], label["h"]
                    heading     = label["y"]
                    category    = label["classId"]

                    if use_custom:
                        category_list = ["Car", "Person", "Bike"]
                        category_list_map = {"Car": "Car", "Person": "Pedestrian", "Bike": "Cyclist"}
                        if category not in category_list:
                            continue
                        else:
                            category= category_list_map[category]
                    else:
                        category = CLASS_REMAP[category]
                    label_str   = " ".join(map(str, [x, y, z, dx, dy, dz, heading, category]))
                    cust_txt.write(label_str+"\n")

                print("Wrote to label file %s, closing..."%cust_path)
                cust_txt.close()

def generate_points(in_root, out_root, trajectories, args, use_custom=False):
    for traj in trajectories:
        print("Generating labels for trajectory %d"%traj)
        meta_path = os.path.join(in_root, "metadata", "%s.json"%traj)
        assert os.path.isfile(meta_path), '%s does not exist' % meta_path

        meta_json = json.load(open(meta_path, "r"))
        split_keys = ["train", "val", "test"]

        for split in split_keys:
            split_dir = os.path.join(out_root, split, "os1")
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

            split_paths = meta_json["ObjectTracking"][split]
            for label_subpath in split_paths:
                label_path = os.path.join(in_root, label_subpath)
                bin_path = label_path.replace("bbox", "raw").replace(".json", ".bin")

                bin_file = bin_path.split("/")[-1]
                frame = int(bin_file.split("_")[-1].split(".")[0])
                #Convert .bin to .npy
                pc_np   = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

                #Downsample from 128 to 64 channels
                vert_ds = 128 // int(args.channels)
                pc_ds_np    = pc_np[:, :4].reshape(1024, 128, 4)
                pc_ds_np    = pc_ds_np[:, np.arange(0, 128, vert_ds), :]
                pc_intensity= pc_ds_np[:, :, -1].reshape(-1, 1)
                pc_ds_np    = pc_ds_np[:, :, :3].reshape(-1, 3)

                #Filter out points in same FOV Velodyne
                pc_dist     = np.linalg.norm(pc_ds_np[:, :3], axis=1)
                zero_mask   = pc_dist!=0
                pc_ds_np    = pc_ds_np[zero_mask]
                pc_intensity= pc_intensity[zero_mask]

                pc_angle    = np.arcsin(pc_ds_np[:, 2] / pc_dist[zero_mask])
                fov_mask    = np.abs(pc_angle) <= 0.2338741
                pc_ds_np    = pc_ds_np[fov_mask, :]
                pc_intensity= pc_intensity[fov_mask, :]

                #Shift point cloud down to match KITTI
                pc_ds_np[:, 2] -= 1.2
                pc_np[:, 2] -= 1.2

                # PCD viewing
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pc_ds_np)
                # o3d.io.write_point_cloud("/home/arthur/AMRL/tmp/%06d_ds.pcd"%frame, pcd, write_ascii=False)

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
                # o3d.io.write_point_cloud("/home/arthur/AMRL/tmp/%06d.pcd"%frame, pcd, write_ascii=False)

                if use_custom:
                    npy_file_dir = os.path.join(out_root, "points")
                    npy_file = "%06d.npy"%frame
                    npy_path = os.path.join(npy_file_dir, npy_file)
                else:
                    npy_file = bin_file.replace(".bin", ".npy")
                    npy_path = os.path.join(split_dir, npy_file)

                np.save(npy_path, np.hstack( (pc_ds_np, pc_intensity) ))
                print("Wrote bin to npy format %s..."%npy_path)

def generate_imagesets(in_root, out_root, trajectories, use_custom=False):
    # Delete old ImageSet Files
    imageset_dir = os.path.join(out_root, "ImageSets")
    
    if os.path.exists(imageset_dir):
        shutil.rmtree(imageset_dir)
        print("Removed existing imageset directory at %s, rebuilding..."%imageset_dir)
    os.makedirs(imageset_dir)

    for traj in trajectories:
        print("Generating imagesets for trajectory %d"%traj)
        meta_path = os.path.join(in_root, "metadata", "%s.json"%traj)
        assert os.path.isfile(meta_path), '%s does not exist' % meta_path

        meta_json = json.load(open(meta_path, "r"))
        split_keys = ["train", "val", "test"]

        for split in split_keys:
            split_dir = os.path.join(out_root, split, "os1")
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

            imageset_path = os.path.join(imageset_dir, "%s.txt"%split)
            imageset_file = open(imageset_path, "a")
            split_paths = meta_json["ObjectTracking"][split]
            for label_subpath in split_paths:
                label_path = os.path.join(in_root, label_subpath)

                label_file = label_path.split("/")[-1]
                label_prefix = label_file.split(".")[0]

                imageset_file.write(label_prefix+'\n')

            imageset_file.close()

def main(args):
    DATASET_ROOT = "/home/arthur/AMRL/Datasets/CODa"
    DATASET_OUT = "/home/arthur/AMRL/Benchmarks/OpenPCDet/data/%s_channel/coda" % str(args.channels)
    if int(args.channels) < 16 or int(args.channels) > 128:
        print("Number of args %s is out of range, exiting..." % str(args.channels))
        exit(0)
    # DATASET_ROOT = "/robodata/arthurz/CODa"
    # DATASET_OUT = "/home/arthur/Benchmarks/OpenPCDet/data/custom"
    TRAJECTORIES    = [2, 3]

    # File Checking
    assert os.path.isdir(DATASET_ROOT), '%s is not a valid dir' % DATASET_ROOT
    if not os.path.exists(DATASET_OUT):
        print("Dataset out dir %s does not exist, creating now..." % DATASET_OUT)
        os.makedirs(DATASET_OUT)

    #Generate Label Files
    generate_labels(DATASET_ROOT, DATASET_OUT, TRAJECTORIES)

    #Generate Points Files
    generate_points(DATASET_ROOT, DATASET_OUT, TRAJECTORIES, args)

    #Generate ImageSets
    generate_imagesets(DATASET_ROOT, DATASET_OUT, TRAJECTORIES)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
