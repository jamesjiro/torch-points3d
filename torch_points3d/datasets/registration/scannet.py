import multiprocessing
import os
import os.path as osp
import random
import struct
import zlib
from itertools import product, repeat

import cv2
import imageio
import numpy as np
import torch
from torch_geometric.data import Data
from torch_points3d.datasets.registration.base_siamese_dataset import (
    BaseSiameseDataset, GeneralFragment)
from torch_points3d.datasets.registration.pair import MultiScalePair, Pair
from torch_points3d.datasets.registration.utils import (
    compute_overlap_and_matches, rgbd2fragment_fine, tracked_matches)
from torch_points3d.datasets.segmentation.scannet import Scannet
from torch_points3d.metrics.registration_tracker import \
    FragmentRegistrationTracker
from torch_points_kernels.points_cpu import ball_query

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}


class RGBDFrame():
    r"""A class for extracting an RGB-D frame from a .sens file in ScanNet.

    Copyright 2017
    Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser,
    Matthias Niessner

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in the
    Software without restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = ''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = ''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:
    r"""A class for saving RGB images, camera intrinsics, and camera poses from a .sens
    file in ScanNet.

    Copyright 2017
    Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser,
    Matthias Niessner

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in the
    Software without restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

  def __init__(self, filename):
    self.version = 4
    self.load(filename)


  def load(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)


  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      if os.path.exists((os.path.join(output_path, str(f) + '.png'))):
        continue
      if f % 100 == 0:
        print('exporting', f, 'th depth frames to', os.path.join(output_path, str(f) + '.png'))

      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)


  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      if os.path.exists((os.path.join(output_path, str(f) + '.png'))):
        continue
      if f % 100 == 0:
        print('exporting', f, 'th color frames to', os.path.join(output_path, str(f) + '.png'))
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      # imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)
      imageio.imwrite(os.path.join(output_path, str(f) + '.png'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting camera intrinsics to', output_path)
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))


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
                           normalize_rgb,
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
    def read_one_scan(scannet_dir,
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
        sens_data.export_depth_images(frame_skip=self.frame_skip)
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
