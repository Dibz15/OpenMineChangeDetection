# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""
from collections.abc import Sequence
from typing import Callable, Optional, Union, Tuple, List

import numpy as np
import torch
from torch import Tensor
from PIL import Image
import os

from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets import OSCD
from torchvision.transforms import Normalize
from .transforms import NormalizeImageDict, NormalizeScale

class OSCD_Chipped(OSCD):

    normalisation_map = {
        "rgb": ([0.1212, 0.1279, 0.1288], [0.2469, 0.2357, 0.1856]),
        "all": ([0.1443, 0.1212, 0.1553, 0.1946, 0.1357, 0.0936, 0.1448, 0.1288, 0.1293,
         0.1907, 0.2039, 0.0324, 0.1620], [0.1377, 0.2469, 0.2937, 0.3299, 0.4074, 0.4604, 0.2402, 0.1856, 0.2878,
         0.3034, 0.3351, 0.1544, 0.3753])
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
        stride: Union[int, Tuple[int, int], List[int]] = 128,
        tile_size: Union[int, Tuple[int, int], List[int]] = 256
    ) -> None:
        super().__init__(root, split, bands, transforms, download, checksum)
        if isinstance(tile_size, int):
            self.tile_shape = (tile_size, tile_size)
        elif isinstance(tile_size, (tuple, list)):
            assert len(tile_size) == 2
            self.tile_shape = tile_size.copy()

        if isinstance(stride, int):
            stride_shape = (stride, stride)
        elif isinstance(stride, (tuple, list)):
            assert len(stride) == 2
            stride_shape = stride.copy()
        
        self.tile_overlap = tuple(max(tile_size_dim - stride_dim, 0) for tile_size_dim, stride_dim in zip(self.tile_shape, stride_shape))
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
                tile_shape=(raw_img_tensor.shape[0], self.tile_shape[0], self.tile_shape[1]),
                overlap=(raw_img_tensor.shape[0]-1, self.tile_overlap[0], self.tile_overlap[1]),
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
              tile_shape=(full_image_shape[0], self.tile_shape[0], self.tile_shape[1]),
              overlap=(full_image_shape[0]-1, self.tile_overlap[0], self.tile_overlap[1]),
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
            sample = self.transforms(sample)

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

    def get_normalization_values(self):
        return OSCD_Chipped.GetNormalizationValues(self.bands)

    @staticmethod
    def GetNormalizationValues(bands="rgb"):
        return OSCD_Chipped.normalisation_map[bands]

    @staticmethod
    def CalcMeanVar(root, split="train", bands="rgb", tile_size: Union[int, Tuple[int, int], List[int]] = 256):
        if isinstance(tile_size, int):
            tile_shape = (tile_size, tile_size)
        elif isinstance(tile_size, (tuple, list)):
            assert len(tile_size) == 2
            tile_shape = tile_size.copy()

        def t(img_dict):
            return {'image': img_dict['image'].to(torch.float) / 10000, 'mask': img_dict['mask']}

        dataset = OSCD(root=root, split=split, bands=bands, download=False, transforms = t)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        def preproc_img(img_dict):
            images = img_dict['image']
            batch_samples = images.size(0)
            n_channels = images.size(1)
            # Separate the tensor into two tensors of shape (B, 3, W, H)
            image1 = images[:, :(n_channels//2), :, :]
            image2 = images[:, (n_channels//2):, :, :]
            # Stack them to get a tensor of shape (2B, 3, W, H)
            images = torch.cat((image1, image2), dim=0)
            images = images.view(batch_samples, images.size(1), -1)
            return images

        mean = 0.0
        for img_dict in loader:
            images = preproc_img(img_dict)
            mean += images.mean(2).sum(0)
        mean = mean / len(loader.dataset)

        var = 0.0
        for img_dict in loader:
            images = preproc_img(img_dict)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
        std = torch.sqrt(var / (len(loader.dataset) * tile_shape[0] * tile_shape[1]))

        return mean, std

    @staticmethod
    def GetScaleTransform():
        return NormalizeScale(scale_factor=10000)
    
    @staticmethod
    def GetNormalizeTransform(bands="rgb"):
        # Mean/STD as 3 or 13 channels/bands
        normalisation = OSCD_Chipped.GetNormalizationValues(bands)
        # We are loading our dataset as a stack of two images (pre/post) as 2 *
        # the number of channels in one image. So for RGB, our tensor has 6
        # channels instead of 2. So we need to stack our normalisation tensor
        # so it matches the image tensor.
        mean, std = torch.tensor(normalisation[0]), torch.tensor(normalisation[1])
        mean, std = torch.cat((mean, mean), dim=0), torch.cat((std, std), dim=0)
        return NormalizeImageDict(mean=mean, std=std)