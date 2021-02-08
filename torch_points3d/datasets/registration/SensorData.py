# Copyright 2017
# Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser,
# Matthias Niessner

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in the
# Software without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os, struct
import numpy as np
import zlib
import imageio

from PIL import Image

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

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
       raise Exception(f'Invalid compression type: {compression_type}')


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise Exception(f'Invalid compression type: {compression_type}')


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)


  def load(self, filename):
    """Loads RGB-D frames into a list from sensor data.

    Args:
        filename (strin): Path to `.sens` file.
    """
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = ''.join(struct.unpack('c'*strlen, f.read(strlen)))
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
      for _ in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)


  def export_depth_images(self, output_path, image_size=None, frame_skip=1, num_fuse=1):
    """Exports depth images from sensor data.

    Args:
        output_path (strin): Path to export images to.
        image_size ([int, int], optional): Size of the images to be exported.
          Defaults to None.
        frame_skip (int, optional): Interval of frames between each export. 
          Defaults to 1.
        num_fuse (int, optional): Number of contiguous images exported every 
          `frame_skip` frames. Defaults to 1.
    """
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print(f'exporting {len(self.frames)//frame_skip} depth frames to {output_path}')
    num_frames = len(self.frames)
    for idx in range(0, num_frames, frame_skip):
      stop = (num_fuse if idx + num_fuse <= num_frames 
              else num_frames)
      for frame in self.frames[idx:stop]: 
        depth_data = frame.decompress_depth(self.depth_compression_type)
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
        if image_size is not None:
          depth = depth.resize((image_size[1], image_size[0]), Image.NEAREST)
        imageio.imwrite(os.path.join(output_path, str(idx) + '.png'), depth)


  def export_color_images(self, output_path, image_size=None, frame_skip=1, num_fuse=1):
    """Exports RGB-D frames from sensor data.

    Args:
        output_path (string): Path to export images to.
        image_size ([int, int], optional): Size of the images being exported. 
          Defaults to None.
        frame_skip (int, optional): Interval of frames between each export. 
          Defaults to 1.
        num_fuse (int, optional): Number of contiguous images exported every 
          `frame_skip` frames. Defaults to 1.
    """
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print(f'exporting {len(self.frames)//frame_skip} color frames to {output_path}')
    num_frames = len(self.frames)
    for idx in range(0, num_frames, frame_skip):
      stop = (num_fuse if idx + num_fuse <= num_frames 
              else num_frames)
      for frame in self.frames[idx:stop]: 
        color = frame.decompress_color(self.color_compression_type)
        if image_size is not None:
          color = color.resize((image_size[1], image_size[0]), Image.NEAREST)
        imageio.imwrite(os.path.join(output_path, str(idx) + '.jpg'), color)


  def save_mat_to_file(self, matrix, filename):
    """Save matrix as a `.txt` file.

    Args:
        matrix (ndarray): Array to be exported.
        filename (string): File to be written.
    """
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1, num_fuse=1):
    """Exports camera extrinics for each frame from sensor data.

    Args:
        output_path (string): Path to export poses to.
        frame_skip (int, optional): Interval of frames between each export. 
          Defaults to 1.
        num_fuse (int, optional): Number of contiguous poses exported every
          `frame_skip` frames. Defaults to 1.
    """
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print(f'exporting {len(self.frames)//frame_skip} camera poses to {output_path}')
    num_frames = len(self.frames)
    for idx in range(0, num_frames, frame_skip):
      stop = (num_fuse if idx + num_fuse <= num_frames 
              else num_frames)
      for frame in self.frames[idx:stop]:
        self.save_mat_to_file(frame.camera_to_world, os.path.join(output_path, str(idx) + '.txt'))


  def export_intrinsics(self, output_path):
    """Exports camera intrinsics from sensor data.

    Args:
        output_path (string): Path to export intrinsics to.
    """
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print(f'exporting camera intrinsics to {output_path}')
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))