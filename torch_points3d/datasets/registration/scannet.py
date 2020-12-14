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


class ScannetRegistration(Scannet, GeneralFragment):
    r"""The ScanNet RGB-D video dataset from the paper
    `"ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes"
    <https://arxiv.org/abs/1702.04405>`_, containing 2.5M views in 1513 scenes
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
        Scannet.__init__(self,
                           root,
                           split,
                           transform,
                           pre_transform,
                           pre_filter,
                           version,
                           max_num_point,
                           process_workers,
                           types,
                           is_test)
        self.frame_skip = frame_skip

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
    def read_one_scan(self, 
                      scannet_dir,
                      scan_name,
                      max_num_points,
                      normalize_rgb):
        """Reads a scan from downloaded files and stores it in an object.

        Args:
            scannet_dir (str): A path to the folder containing the raw scan data
            scan_name (str): The name of the scan
            max_num_points (int): The maximum number of points sampled from the scan
            normalize_rgb (bool): Normalize RGB values

        Returns:
            data (Data): A PyTorch Geometric Data object containing the scan data.
        """
        sens_filename = osp.join(scannet_dir, scan_name, scan_name + ".sens")
        sens_data = SensorData(sens_filename)
        # TODO: Add output_path
        # sens_data.export_depth_images(frame_skip=self.frame_skip)
        # TODO: Save to new frames_dir
        return

    # TODO
    @property
    def processed_file_names(self):
        pass

    def process(self):
        super().process()

    def download(self):
        super().download()
