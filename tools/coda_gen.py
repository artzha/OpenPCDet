import os
import pdb
import json

import numpy as np

def generate_labels(in_root, out_root, trajectories):
    label_set = {str(traj): [] for traj in trajectories}
    category_list = ["Car", "Person", "Bike"]

    for traj in trajectories:
        print("Generating labels for trajectory %d"%traj)
        subdir = os.path.join("3d_label", "os1", str(traj))
        trajdir= os.path.join(in_root, subdir)
        assert os.path.isdir(trajdir), '%s does not exist' % trajdir
        
        cust_file_dir   = os.path.join(out_root, "labels")
        if not os.path.exists(cust_file_dir):
            print("Creating new output label dir at %s"%cust_file_dir)
            os.makedirs(cust_file_dir)

        label_files = [lfile for lfile in sorted(os.listdir(trajdir)) if os.path.isfile(
            os.path.join(trajdir, lfile) ) ] 

        for label_file in label_files:
            label_path = os.path.join(trajdir, label_file)

            label_json = json.load(open(label_path, "r"))
            label_list = label_json["3dannotations"]

            frame = int(label_file.split("_")[-1].split(".")[0])
            label_set[str(traj)].append(frame)
            #Create output label file
            cust_file   = "%06d.txt" % frame
            cust_path       = os.path.join(cust_file_dir, cust_file)
            
            cust_txt    = open(cust_path, "w+")
            for label in label_list:
                #Write to output
                x, y, z     = label["cX"], label['cY'], label['cZ']
                dx, dy, dz  = label["l"]/2, label["w"]/2, label["h"]/2
                heading     = label["y"]
                category    = label["classId"]
                if category not in category_list:
                    continue
                label_str   = " ".join(map(str, [x, y, z, dx, dy, dz, heading, category]))
                cust_txt.write(label_str+"\n")

            print("Wrote to label file %s, closing..."%cust_path)
            cust_txt.close()

    return label_set

def generate_points(in_root, out_root, trajectories, label_set):

    for traj in trajectories:
        print("Generating points for trajectory %d"%traj)
        subdir = os.path.join("3d_raw", "os1", str(traj))
        trajdir= os.path.join(in_root, subdir)
        assert os.path.isdir(trajdir), '%s does not exist' % trajdir
        
        npy_file_dir = os.path.join(out_root, "points")
        if not os.path.exists(npy_file_dir):
            print("Creating new output points dir at %s"%npy_file_dir)
            os.makedirs(npy_file_dir)

        bin_files = [lfile for lfile in sorted(os.listdir(trajdir)) if os.path.isfile(
            os.path.join(trajdir, lfile) ) ] 
        for bin_file in bin_files:
            frame = int(bin_file.split("_")[-1].split(".")[0])
            if frame not in label_set[str(traj)]:
                continue
            #Convert .bin to .npy
            bin_path = os.path.join(trajdir, bin_file)
            pc_np   = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

            npy_file = "%06d.npy"%frame
            npy_path = os.path.join(npy_file_dir, npy_file)
            np.save(npy_path, pc_np)
            print("Wrote bin to npy format %s..."%npy_path)

def generate_imagesets(in_root, out_root, trajectories):
    # Delete old ImageSet Files
    imageset_dir = os.path.join(out_root, "ImageSets")
    if not os.path.exists(imageset_dir):
        print("ImageSet directory does not exist at %s, creating..."%imageset_dir)
        os.makedirs(imageset_dir)
    train_path  = os.path.join(imageset_dir, "train.txt")
    val_path    = os.path.join(imageset_dir, "val.txt")
    test_path   = os.path.join(imageset_dir, "test.txt")
    train_file  = open(train_path, "w+")
    val_file    = open(val_path, "w+")
    test_file   = open(test_path, "w+")

    # Create New ImageSet Files
    for traj in trajectories:
        print("Generating imagesets for trajectory %d"%traj)
        subdir = os.path.join("3d_label", "os1", str(traj))
        trajdir= os.path.join(in_root, subdir)
        assert os.path.isdir(trajdir), '%s does not exist' % trajdir
        
        label_files = [lfile for lfile in sorted(os.listdir(trajdir)) if os.path.isfile(
            os.path.join(trajdir, lfile) ) ] 

        # Allocate first 60%train  20%validation 20%test
        num_train   = int(len(label_files)*0.8)
        num_val     = int(len(label_files)*0) # Use test as val
        num_test    = len(label_files) - num_train - num_val

        # Generate train and val sets
        for (idx, label_file) in enumerate(label_files):
            frame = int(label_file.split("_")[-1].split(".")[0])
            frame_str = "%06d\n"%frame

            if idx < num_train:
                train_file.write(frame_str)
            elif idx < (num_train+num_val):
                val_file.write(frame_str)
            else:
                test_file.write(frame_str)

    train_file.close()
    val_file.close()
    test_file.close()

def main():
    # DATASET_ROOT = "/home/arthur/AMRL/Datasets/CODa"
    DATASET_ROOT = "/robodata/arthurz/CODa"
    DATASET_OUT = "/home/arthurz/Benchmarks/OpenPCDet/data/custom"
    TRAJECTORIES    = [2, 3]

    # File Checking
    assert os.path.isdir(DATASET_ROOT), '%s is not a valid dir' % DATASET_ROOT
    assert os.path.isdir(DATASET_OUT), '%s is not a valid dir' % DATASET_OUT

    #Generate Label Files
    label_set = generate_labels(DATASET_ROOT, DATASET_OUT, TRAJECTORIES)

    #Generate Points Files
    generate_points(DATASET_ROOT, DATASET_OUT, TRAJECTORIES, label_set)

    #Generate ImageSets
    generate_imagesets(DATASET_ROOT, DATASET_OUT, TRAJECTORIES)


if __name__ == '__main__':
    main()
