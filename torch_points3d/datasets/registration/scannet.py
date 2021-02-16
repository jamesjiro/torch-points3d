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
                 min_overlap_ratio=0.3,
                 self_supervised=False,
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
        self.max_dist_overlap = max_distance_overlap
        self.min_overlap_ratio = min_overlap_ratio
        self.self_supervised = self_supervised
    
    @property
    def raw_file_names(self):
        return ["scans"]

    @property
    def processed_file_names(self):
        return ["raw_fragment", "fragment", "pair_overlap"]
    
    def _create_fragments(self):
        out_dir = osp.join(self.processed_dir, 'raw_fragment')

        if files_exist(out_dir):
            log.warning("Raw fragments already exist")
            return

        makedirs(out_dir)

        # Iterate over each scene in the raw scans.
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
            out_path = osp.join(out_dir, scene_path)
            makedirs(out_path)
            # Get indices for `num_frame_per_fragment` frames each `frame_skip`.
            frame_inds = [
                base + step 
                for base in range(0, num_frames, self.frame_skip) 
                for step in range(self.num_frame_per_fragment 
                                  if base + self.num_frame_per_fragment < num_frames
                                  else num_frames - base)
            ]
            # Get lists of paths for frames and poses.
            list_path_frames = [osp.join(depth, "{}.png".format(ind)) 
                                for ind in frame_inds]
            list_path_trans = [osp.join(pose, "{}.txt".format(ind))
                                for ind in frame_inds]
            # Fuse `num_frame_per_fragment` frames with camera poses and intrinsics.
            # Save fragment for every `frame_skip` frames.
            rgbd2fragment_fine(list_path_frames, path_intrinsic, 
                                list_path_trans, out_path, 
                                self.num_frame_per_fragment,
                                self.voxel_size, 
                                pre_transform=None,
                                depth_thresh=self.depth_thresh,
                                save_pc=True,
                                limit_size=self.limit_size)
    
    # TODO
    def _pre_transform_fragment(self):
        """
        Apply `pre_transform` to `raw_fragments` and save to `fragments`.
        """
        out_dir = osp.join(self.processed_dir, 'fragment')

        if files_exist([out_dir]):
            log.warning("Pre-transformed fragments already exist.")
            return
        
        makedirs(out_dir)

        # for scene_path in os.listdir(osp.join(self.raw_dir)):
        raise NotImplementedError("Implement process method")

    def _compute_fragment_pairs(self):
        raw_pair_path = osp.join(self.processed_dir, 'pair_overlap')

        if files_exist(raw_pair_path):
            log.warning("Pair overlap already computed")
            return

        makedirs(raw_pair_path)

        for scene_path in os.listdir(osp.join(self.processed_dir, 'fragment')):
            num_fragments = len(os.listdir(scene_path))
            log.info("{}, num_fragments: {}".format(scene_path, num_fragments))

            frag_idx = 0
            pair_idx = 0

            while (frag_idx < num_fragments - 1):
                out_path = osp.join(raw_pair_path, 'pair{:06}.npy'.format(pair_idx))
                path1 = "fragment_{:06d}.pt".format(frag_idx)
                path2 = "fragment_{:06d}.pt".format(frag_idx+1)
                data1 = torch.load(path1)
                data2 = torch.load(path2)
                match = compute_overlap_and_matches(
                    data1, data2, self.max_dist_overlap)
                match['path_source'] = path1
                match['path_target'] = path2

                if match['overlap'] >= self.min_overlap_ratio:
                    np.save(out_path, match)
                    frag_idx += 2
                    pair_idx += 1
                else:
                    frag_idx += 1

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
    
    # TODO
    def process(self):
        log.info("create fragments")
        self._create_fragments()
        log.info("apply pre_transform to fragments")
        self._pre_transform_fragment()
        log.info("compute pairs")
        self._compute_fragment_pairs()
        raise NotImplementedError("Implement process method")

    # TODO: check what idx is indexing
    def __getitem__(self, idx):
        res = self.get_fragment(idx)
        return res

    def download(self):
        super().download()
