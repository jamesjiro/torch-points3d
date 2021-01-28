import logging
import multiprocessing
import os
import os.path as osp
import random
from itertools import product, repeat

import imageio
import numpy as np
import torch
from torch_geometric.data import Data
from torch_points3d.datasets.registration.base_siamese_dataset import (
    BaseSiameseDataset, GeneralFragment)
from torch_points3d.datasets.registration.pair import MultiScalePair, Pair
from torch_points3d.datasets.registration.SensorData import RGBDFrame, SensorData
from torch_points3d.datasets.registration.utils import (
    compute_overlap_and_matches, rgbd2fragment_fine, tracked_matches)
from torch_points3d.datasets.segmentation.scannet import Scannet
from torch_points3d.metrics.registration_tracker import \
    FragmentRegistrationTracker

log = logging.getLogger(__name__)


class ScannetRegistration(Scannet, GeneralFragment):
    r"""The ScanNet RGB-D video dataset from the paper
    `"ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes"
    <https://arxiv.org/abs/1702.04405>`, containing 2.5M views in 1513 scenes
    annotated with 3D camera poses, surface reconstructions and semantic
    segmentations.

    Applied to registration.

    Args:
        root (string): 
            The root directory where the dataset should be saved.
        frame_skip (int, optional):
            The number of frames between each RGB-D scan sampled from the raw ScanNet
            videos. Defaults to 25.
        split (str, optional):
            The dataset split used(train, val or test). Defaults to "train".
        transform (callable, optional): 
            A function/transform that takes in an :obj:`torch_geometric.data.Data`
            object and returns a transformed version. The data object will be
            transformed before every access. Defaults to None.
        pre_transform (callable, optional): 
            A function/transform that takes in an :obj:`torch_geometric.data.Data`.
            object and returns a transformed version. The data object will be
            transformed before being saved to disk. Defaults to None.
        pre_filter (callable, optional): 
            A function that takes an :obj:`torch_geometric.data.Data` object and
            returns a boolean value, indicating whether the data object should be
            included in the final dataset. Defaults to None.
        version (str, optional): 
            The version of ScanNet. Defaults to "v1".
        max_num_point (int, optional): 
            The maximum number of points to keep during the preprocessing step. 
            Defaults to None.
        process_workers (int, optional): 
            The number of workers for processing. Defaults to 4.
        types (list, optional): 
            The types of data to use get from the ScanNet dataset. 
            Defaults to [".sens"].
        is_test (bool, optional): [description]. Defaults to False.
    """
    
    def __init__(self,
                 root,
                 frame_skip=25,
                 split="train",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 version="v1",
                 max_num_point=None,
                 process_workers=4,
                 types=[".sens"],
                 is_test=False):
        Scannet.__init__(self, root, split, transform, pre_transform, pre_filter, 
                         version, max_num_point, process_workers, types, is_test)
        self.frame_skip = frame_skip
        assert version == "v1", "The version should be v1"
        assert split == "train", "The data split should be `train`"
        self.version = version
        self.max_num_point = max_num_point
        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers
        self.is_test = is_test
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    def get_raw_pair(self, idx):
        data_source_o = self.get_raw_data(idx)
        data_target_o = self.get_raw_data(idx)
        data_source, data_target, new_pair = self.unsupervised_preprocess(
            data_source_o, data_target_o)
        return data_source, data_target, new_pair

    def __getitem__(self, idx):
        res = self.get_fragment(idx)
        return res

    @staticmethod
    def read_one_scan(scannet_dir, scan_name):
        """Reads a scan from downloaded .sens files and exports RGB-D frames, camera
        intrinsics, and poses.

        Args:
            scannet_dir (str): A path to the folder containing the raw scan data
            scan_name (str): The name of the scan
        """
        sens_filename = osp.join(scannet_dir, scan_name, scan_name + ".sens")
        output_path = osp.join(scannet_dir, scan_name)
        # Read raw .sens file and export RGB-D images, poses and intrinsics
        paths = {x: osp.join(output_path, x) for x in ['depth', 'pose', 'intrinsic']}
        sd = SensorData(sens_filename)
        sd.export_depth_images(paths['depth'], frame_skip=self.frame_skip)
        sd.export_poses(paths['pose'], frame_skip=self.frame_skip)
        sd.export_intrinsics(paths['intrinsic'])

    @staticmethod
    def process_func(id_scan, total, scannet_dir, scan_name):
        ScannetRegistration.read_one_scan(scannet_dir, scan_name)
        log.info("{}/{}| scan_name: {}".format(id_scan, total, scan_name))

    # TODO
    def process(self):
        pass

    # TODO: Get Point clouds from RGB-D images
    # file_paths = {x: [osp.join(paths[x], f) for f in os.listdir(paths[x])]
    #               for x in ['depth', 'pose']}
    # path_intrinsic = osp.join(paths['intrinsic'], 'intrinsic_depth.txt')
    # path_fragment = osp.join(output_path, 'fragment')

    # os.makedirs(path_fragment)
    # # Fuse RGB-D frames with TSDF volume and save point cloud fragement
    # for i in range(len(file_paths)):
    #     image = file_paths['depth'][i : i + 1]
    #     pose = file_paths['pose'][i : i + 1]
    #     rgbd2fragment_fine(image, path_intrinsic, pose, path_fragment, num_frame_per_fragment=1)
    # TODO: Get overlapping point cloud pair
      
    # TODO
    @property
    def processed_file_names(self):
        pass

    def download(self):
        super().download()
