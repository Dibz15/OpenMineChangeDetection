# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""

import glob
import os
from collections.abc import Sequence
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets import OSCD
from torchgeo.datasets.utils import (
    download_url,
    draw_semantic_segmentation_masks,
    extract_archive,
    sort_sentinel2_bands,
)

from functools import reduce
import operator

class OSCD_Chipped(OSCD):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
        stride: int = 128
    ) -> None:
        super().__init__(root, split, bands, transforms, download, checksum)
        self.stride = stride
        self.total_dataset_length, self.chip_index_map, self.image_shapes_map = self._calculate_dataset_len()

    def _load_raw_index(self, index: int):
        files = self.files[index]
        image1 = self._load_image(files["images1"])
        image2 = self._load_image(files["images2"])
        mask = self._load_target(str(files["mask"]))
        return image1, image2, mask

    def _load_tensor_index(self, index: int):
        image1, image2, mask = self._load_raw_index(index)
        image1_tensor = torch.from_numpy(image1)
        image2_tensor = torch.from_numpy(image2)
        mask_tensor = torch.from_numpy(mask).to(torch.long)
        raw_img_tensor = torch.cat([image1_tensor, image2_tensor])
        return raw_img_tensor, mask_tensor

    def _calculate_dataset_len(self):
      """Returns the total length (number of chips) of the dataset
          This is the total number of tiles after tiling every high-res image
          in the dataset and calculating the tiling using the tiler.

          This function also creates and returns a dictionary that maps from the
          chipped dataset index to the original file index.

      Returns:
          - Length of the dataset in number of chips
          - Index map (key is the chip index, value is a tuple of (file index, chip index relative to the last file))
          - Map of image shapes
      """
      total_tiles = 0
      index_map = {}
      image_shapes = {}

      for i in range(len(self.files)):
          files = self.files[i]
          image1 = self._load_image(files["images1"])
          image1_tensor = torch.from_numpy(image1)
          raw_img_tensor = torch.cat([image1_tensor, image1_tensor])
          
          tiler = Tiler(data_shape=raw_img_tensor.shape,
                tile_shape=(raw_img_tensor.shape[0], 256, 256),
                overlap=max(256 - self.stride, 0),
                mode="drop",
                channel_dimension=0)
          
          image_shapes[i] = list(raw_img_tensor.shape)
          tile_shape = tiler.get_mosaic_shape(with_channel_dim=True)
          num_tiles = tile_shape[0] * tile_shape[1] * tile_shape[2]
          # Map chip index to file index
          for j in range(total_tiles, total_tiles + num_tiles):
              index_map[j] = (i, j - total_tiles)

          total_tiles += num_tiles

      return total_tiles, index_map, image_shapes

    def __getitem__(self, chip_index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            chip_index: index to return

        Returns:
            data and label at that index
        """

        (file_index, relative_chip_index) = self.chip_index_map[chip_index]
        full_image_shape = self.image_shapes_map[file_index].copy()
        full_image_shape[0] = full_image_shape[0] + 1
        tiler = Tiler(data_shape=full_image_shape,
              tile_shape=(full_image_shape[0], 256, 256),
              overlap=max(256 - self.stride, 0),
              mode="drop",
              channel_dimension=0)

        # img, mask = self._load_tensor_index(file_index)
        img1, img2, mask = self._load_raw_index(file_index)
        mask = np.expand_dims(mask, 0)
        img_full = np.concatenate((img1, img2, mask))
        
        img_mask_tile = tiler.get_tile(img_full, relative_chip_index, copy_data = False)

        img_tile, mask_tile = img_mask_tile[0:full_image_shape[0] - 1, :, :], img_mask_tile[full_image_shape[0] - 1, :, :]
        img_tile_tensor = torch.from_numpy(img_tile)
        mask_tile_tensor = torch.from_numpy(mask_tile).to(torch.long)

        # image = torch.cat([image1, image2])
        sample = {"image": img_tile_tensor, "mask": mask_tile_tensor}

        if self.transforms is not None:
            sample = self.transforms(sample['image'])

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return self.total_dataset_length

    def _load_image(self, paths: Sequence[str]) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        images: list["np.typing.NDArray[np.int_]"] = []
        for path in paths:
            with Image.open(path) as img:
                images.append(np.array(img))
        array: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0).astype(np.int_)
        return array

    def _load_target(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            array = np.clip(array, a_min=0, a_max=1).astype(np.int_)
            return array