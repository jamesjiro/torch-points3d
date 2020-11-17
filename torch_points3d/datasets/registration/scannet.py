import os
import os.path as osp
import random
from itertools import product, repeat

import numpy as np
import torch

from torch_geometric.data import Data
from torch_points3d.datasets.segmentation.scannet import Scannet
from torch_points3d.datasets.registration.base_siamese_dataset import (
    BaseSiameseDataset, GeneralFragment)
from torch_points3d.datasets.registration.pair import MultiScalePair, Pair
from torch_points3d.datasets.registration.utils import (
    compute_overlap_and_matches, tracked_matches)
from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker
from torch_points_kernels.points_cpu import ball_query


class ScannetRegistration(Scannet, GeneralFragment):
    r"""The ScanNet RGB-D video dataset from the paper 
    `"ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes"
    <https://arxiv.org/abs/1702.04405>`_, containing 2.5M views in 1513 scenes
    annotated with 3D camera poses, surface reconstructions and semantic 
    segmentations.

    Applied to registration.
    """
    def __init__(self, root, *args, **kwargs):
        Scannet().__init__(self,
                           root,
                           split="train",
                           transform=None,
                           pre_transform=None,
                           pre_filter=None,
                           version="v1",
                           max_num_point=None,
                           process_workers=4)

    def get_raw_pair(self, idx):
        data_source_o = self.get_model(idx)
        data_target_o = self.get_model(idx)
        data_source, data_target, new_pair = self.unsupervised_preprocess(
            data_source_o, data_target_o)
        return data_source, data_target, new_pair

    def __getitem__(self, idx):
        res = self.get_fragment(idx)
        return res

    def get_name(self, idx):
        data = self.get_model(idx)
        return data.y.item(), "{}_source".format(idx), "{}_target".format(idx)

    def process(self):
        super().process()

    def download(self):
        super().download()
