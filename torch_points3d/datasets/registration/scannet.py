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
                 num_fuse=1,
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
        scannet_dir = osp.join(self.raw_dir, "scans")
        scene_paths = [osp.join(scannet_dir, scene) for scene in os.listdir(scannet_dir)]
        self.scene_paths = scene_paths        
        self.frame_skip = frame_skip
        self.num_fuse = num_fuse

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
    def _read_raw_scan(scene, frame_skip, num_fuse):
        """Reads a scan from downloaded .sens files and exports `num_fuse` contiguous 
        RGB-D frames, camera intrinsics, and poses every `frame_skip` frames.

        Args:
            scannet_dir (str): A path to the folder containing the raw scan data
            scan_name (str): The name of the scan
        """
        sens_filename = osp.join(scene, scene + ".sens")
        paths = {x: osp.join(scene, x) for x in ['depth', 'pose', 'intrinsic']}
        sd = SensorData(sens_filename)
        sd.export_depth_images(paths['depth'], frame_skip, num_fuse)
        sd.export_poses(paths['pose'], frame_skip, num_fuse)
        sd.export_intrinsics(paths['intrinsic'])
    
    def process_raw_scans(self):
        """Processes raw scans and exports images, camera intrinsics, and camera poses
        for each scene in ScanNet.
        """
        args = [(scene, self.frame_skip) for scene in self.scene_paths]
        if self.use_multiprocessing:
            with multiprocessing.get_context("spawn").Pool(processes=self.process_workers) as pool:
                pool.starmap(self._read_raw_scan, args)
        else:
            for arg in args:
                self._read_raw_scan(*arg)

    def fuse_frames(self):
        num_fuse = self.num_fuse
        for path in self.scene_paths:
            frag = osp.join(path, 'fragment')
            depth = osp.join(path, 'depth')
            pose = osp.join(path, 'pose')
            path_intrinsic = osp.join(path, 'intrinsic')
            num_frames = len(os.listdir(depth))
            for idx in range(num_frames):
                stop = num_fuse if idx + num_fuse <= num_frames else num_frames
                list_paths = [ 
                    (osp.join(depth, "{}.png".format(f)),
                     osp.join(pose, "{}.txt".format(f)))
                    for f in range(idx, stop)
                ]
                rgbd2fragment_fine(list_paths[0], path_intrinsic, list_paths[1],
                                   frag, num_frame_per_fragment=num_fuse)

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
      
    def download(self):
        super().download()
