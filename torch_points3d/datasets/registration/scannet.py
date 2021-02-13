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
from torch_points3d.datasets.registration.SensorData import (RGBDFrame,
                                                             SensorData)
from torch_points3d.datasets.registration.utils import (
    compute_overlap_and_matches, files_exist, makedirs, rgbd2fragment_fine,
    tracked_matches)
from torch_points3d.datasets.segmentation.scannet import Scannet
from torch_points3d.metrics.registration_tracker import \
    FragmentRegistrationTracker

log = logging.getLogger(__name__)


class ScannetRegistration(Scannet, GeneralFragment):
    # TODO: Update DocString
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
                 num_frame_per_fragment=1,
                 tsdf_voxel_size=0.01,
                 limit_size=600,
                 depth_thresh=6,
                 max_distance_overlap=0.05,
                 split="train",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 version="v2",
                 max_num_point=None,
                 process_workers=4,
                 types=[".sens"]):
        Scannet.__init__(self, root, split, transform, pre_transform, pre_filter, 
                         version, max_num_point, process_workers, types)
        self.frame_skip = frame_skip
        self.num_frame_per_fragment = num_frame_per_fragment
        self.tsdf_voxel_size = tsdf_voxel_size
        self.limit_size = limit_size
        self.depth_thresh = depth_thresh
        self.max_distance_overlap = max_distance_overlap
    
    @property
    def raw_file_names(self):
        return ["scans"]

    @property
    def processed_file_names(self):
        return ["fragment", "raw_pair", "pair_overlap"]
    
    def _create_fragment(self):
        fragment_path = osp.join(self.processed_dir, 'fragment')
        if files_exist(fragment_path):
            log.warning("Raw fragments already exist")
            return
        makedirs(fragment_path)
        for scene_path in os.listdir(osp.join(self.raw_dir, "scans")):
            depth = osp.join(scene_path, 'depth')
            pose = osp.join(scene_path, 'pose')
            path_intrinsic = osp.join(scene_path, 'intrinsic')
            num_frames = len(os.listdir(depth))
            assert num_frames == len(os.listdir(pose)), \
                log.error("For the scene {}, "
                          "the number of depth frames "
                          "does not equal the number of " 
                          "camera poses.".format(scene_path))
            out_path = osp.join(fragment_path, scene_path)
            makedirs(out_path)
            for idx in range(0, num_frames, self.num_frame_per_fragment):
                stop = (self.num_frame_per_fragment 
                        if idx + self.num_frame_per_fragment <= num_frames 
                        else num_frames)
                list_path_frames = [osp.join(depth, "{}.png".format(frame)) 
                                    for frame in range(idx, stop)]
                list_path_trans = [osp.join(pose, "{}.txt".format(frame))
                                   for frame in range(idx, stop)]
                rgbd2fragment_fine(list_path_frames, path_intrinsic, 
                                    list_path_trans, out_path, 
                                    self.num_frame_per_fragment,
                                    self.voxel_size, 
                                    pre_transform=None,
                                    depth_thresh=self.depth_thresh,
                                    save_pc=True,
                                    limit_size=self.limit_size)

    def _compute_fragment_pairs(self):
        raw_pair_path = osp.join(self.processed_dir, 'pair_overlap')
        if files_exist(raw_pair_path):
            log.warning("Pair overlap already computed")
            return
        makedirs(raw_pair_path)
        for scene_path in os.listdir(osp.join(self.processed_dir, 'fragment')):
            num_fragments = len(os.listdir(scene_path))
            log.info("{}, num_fragments: {}".format(scene_path, num_fragments))
            idx = 0
            num_pairs = 0
            while (idx < num_fragments - 1):
                out_path = osp.join(raw_pair_path, 'pair{:06}.npy'.format(idx))
                path1 = "fragment_{:06d}.pt".format(idx)
                path2 = "fragment_{:06d}.pt".format(idx+1)
                data1 = torch.load(path1)
                data2 = torch.load(path2)
                match = compute_overlap_and_matches(
                    data1, data2, self.max_distance_overlap)
                # TODO: compute overlap between fragment idx and idx + 1
                # If overlap >= 0.3 then raw_pair = (fragment_idx, fragment_idx+1)
                # else idx = idx + 1
                output = compute_overlap_and_matches(frag1, frag2, 
                                                     self.max_distance_overlap)
                if output['overlap'] >= 0.3:

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
    def _read_raw_scan(path, frame_skip, num_frame_per_fragment):
        """Reads a scan from downloaded .sens files and exports `num_frame_per_fragment`
        contiguous RGB-D frames, camera intrinsics, and poses every `frame_skip` frames.

        Args:
            scannet_dir (str): A path to the folder containing the raw scan data
            scan_name (str): The name of the scan
        """
        sens_filename = osp.join(path, path + ".sens")
        paths = {x: osp.join(path, x) for x in ['depth', 'pose', 'intrinsic']}
        sd = SensorData(sens_filename)
        sd.export_depth_images(paths['depth'], frame_skip, num_frame_per_fragment)
        sd.export_poses(paths['pose'], frame_skip, num_frame_per_fragment)
        sd.export_intrinsics(paths['intrinsic'])
    
    def process_raw_scans(self):
        """Processes raw scans and exports images, camera intrinsics, and camera poses
        for each scene in ScanNet.
        """
        scan_dir = osp.join(self.raw_dir, "scans")
        scene_paths = [osp.join(scan_dir, scene) for scene in os.listdir(scan_dir)]
        args = [(path, self.frame_skip, self.num_frame_per_fragment) 
                for path in scene_paths]
        if self.use_multiprocessing:
            with multiprocessing.get_context("spawn").Pool(processes=self.process_workers) as pool:
                pool.starmap(self._read_raw_scan, args)
        else:
            for arg in args:
                self._read_raw_scan(*arg)

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
