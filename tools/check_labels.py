import argparse
import glob
from pathlib import Path
import os

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = [os.path.join(root_path, "points", file) for file in os.listdir(os.path.join(root_path, "points")) if os.path.isfile(os.path.join(root_path, "points", file))]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.label_file_list = [os.path.join(root_path, "labels", file.split("/")[-1]).replace(".npy", ".txt") for file in data_file_list]

        self.class_label_map = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        
        gt_boxes = np.loadtxt(self.label_file_list[index], dtype=np.float32, usecols=(0, 1, 2, 3, 4, 5, 6))
        gt_labels  = np.loadtxt(self.label_file_list[index], dtype=str, usecols=(7))

        # Added for KITTI offset
        # points[:, 2] -= 1.2
        input_dict = {
            'points': points,
            'frame_id': index,
            'gt_boxes': gt_boxes,
            'gt_names': gt_labels
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['gt_labels'] = [self.class_label_map[label] for label in gt_labels]
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
                        
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            import pdb; pdb.set_trace()
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=data_dict['gt_boxes'][0],
                ref_scores=None, ref_labels=data_dict['gt_labels'][0]
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
