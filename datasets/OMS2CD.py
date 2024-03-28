"""
Author: Austin Dibble

This file includes code derived from the torchgeo project,
which is licensed under the MIT License. The original license notice
is included below:

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""

"""
The majority of the code in this file is not derived from torchgeo, and is licensed
under OSL 3.0, as described in the README.
"""

import csv
import string
import os
import numpy as np
import matplotlib.pyplot as plt
from rasterio import transform
from rasterio.transform import Affine
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union, Tuple, List
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchgeo.datasets import NonGeoDataset, OSCD
from torchgeo.transforms import AugmentationSequential
from torchgeo.datasets.utils import draw_semantic_segmentation_masks
from torchvision.transforms import Normalize
from .transforms import NormalizeScale, NormalizeImageDict, TransformedSubset
from tiler import Tiler, Merger
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datamodules.utils import dataset_split
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms.transforms import _RandomNCrop
import torch
import kornia.augmentation as K
import rasterio

class OMS2CD(NonGeoDataset):
    normalisation_map = {
        "rgb": (torch.tensor([192.9278, 185.3099, 191.5534]), torch.tensor([55.8556, 53.8176, 55.0711])),
        "rgbnir": (torch.tensor([192.9281, 185.3099, 191.5535, 200.3148]), torch.tensor([55.8543, 53.8187, 55.0701, 53.1153])),
        "all": (torch.tensor([1571.1372, 1365.5087, 1284.8223, 1298.9539, 1431.2260, 1860.9531,
                    2081.9634, 1994.7665, 2214.5986,  641.4485,   14.3672, 1957.3165,
                    1419.6107]),
                torch.tensor([274.9591,  414.6901,  537.6539,  765.5303,  724.2261,  760.2133,
                    848.7888,  866.8081,  920.1696,  322.1572,    8.6878, 1019.1249,
                    872.1970])
              )
    }
    mean = normalisation_map['all'][0]
    std = normalisation_map['all'][1]

    colormap = ['blue']
    def __init__(
        self,
        root,
        split: str = "train",
        bands: str = "rgb",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        stride: Union[int, Tuple[int, int], List[int]] = 128,
        tile_size: Union[int, Tuple[int, int], List[int]] = 256,
        tile_mode: str = "drop",
        load_area_mask: bool = False,
        index_no_mask: bool = True
    ) -> None:
        """
        OMS2CD Dataset loader. Extends torchgeo's NonGeoDataset. Dataset is indexed by the tile number. len(dataset) will give the number of tiles in the chosen split, 
        calculated from the given tile_size, tile_mode, and stride.

        Args:
            - root (str): The root directory where the dataset is stored.
            - split (str, optional): The split of the dataset. Should be one of 'train', 'val', 'test', or 'all'.
                                Default is 'train'.
            - bands (str, optional): The bands to use. Should be one of 'rgb', 'rgbnir', or 'all'. Default is 'rgb'.
            - transforms (Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]], optional): Transformations to be applied
                                                                                                to the image pairs. Default is None.
            - stride (Union[int, Tuple[int, int], List[int]], optional): The stride used for creating tiles from images.
                                                                        Default is 128.
            - tile_size (Union[int, Tuple[int, int], List[int]], optional): The size of the tiles. Default is 256.
            - tile_mode (str, optional): The tile mode to use. Should be one of 'drop', or 'constant'. Default is 'drop'. Using 'constant' increases
                                        the effective dataset size but allows incomplete tiles padded by 0.
            - load_area_mask (bool, optional): Whether to load the area mask. Default is False.
            - index_no_mask (bool, optional): Whether to include image pairs with no area mask in the dataset index. Default is True.

        Example:
            dataset = OMS2CD(root='OMS2CD', split='train', bands='rgb')
            sample = dataset[0]
        """
        assert bands in ['rgb', 'rgbnir', 'all']
        assert split in ['train', 'val', 'test', 'all']
        
        self.root_dir = root
        self.bands = bands
        self.split = split
        self.index_no_mask = index_no_mask

        self.mean = self.normalisation_map[bands][0]
        self.std = self.normalisation_map[bands][1]
        self.load_area_mask = load_area_mask
        self.tile_mode = tile_mode

        self.file_list = self._build_index()  # using build_index method to build the file list
        self.transforms = transforms

        if tile_size is None:
            self.no_tile = True
        else:
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
            self.no_tile = False

        self.total_dataset_length, self.chip_index_map, self.image_shapes_map = self._calculate_dataset_len()

    def _get_date_str(self, s2_file):
        s2_file_without_ext = s2_file.replace('.tif', '')
        _, date_str = s2_file_without_ext.rsplit('_', 1)
        return date_str

    def _build_index(self):
        valid_facilities = set()
        if self.split != 'all':
            split_file_path = os.path.join(self.root_dir, f'{self.split}.csv')
            with open(split_file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                valid_facilities = set([str(row[0]) for row in reader])
        index_list = []
        facilities_list = set()
        with open(os.path.join(self.root_dir, 'mapping.csv'), 'r', newline='') as mapping_file:
            reader = csv.DictReader(mapping_file)
            for row in reader:
                id = row['id']
                facility = row['mask'].replace('.tif', '')
                if len(valid_facilities) and facility not in valid_facilities:
                    continue
                facilities_list.add(facility)
                imageA_path = os.path.join(self.root_dir, row['imageA'])
                imageB_path = os.path.join(self.root_dir, row['imageB'])
                mask_path = os.path.join(self.root_dir, 'mask', f'{facility}_{id}.tif')
                area_mask_path = os.path.join(self.root_dir, 'area_mask', row['mask'])

                predate, postdate = self._get_date_str(row['imageA']), self._get_date_str(row['imageB'])

                if self.load_area_mask:
                    if os.path.isfile(imageA_path) and os.path.isfile(imageB_path) \
                            and os.path.isfile(mask_path) and os.path.isfile(area_mask_path):
                        index_list.append((imageA_path, imageB_path, mask_path, area_mask_path, facility, predate, postdate, id))
                else:
                    if os.path.isfile(imageA_path) and os.path.isfile(imageB_path) \
                            and os.path.isfile(mask_path):
                        index_list.append((imageA_path, imageB_path, mask_path, area_mask_path, facility, predate, postdate, id))

        self.facilities_list = facilities_list
        return index_list

    def _load_raw_index(self, index: int):
        files = self.file_list[index]
        imageA, meta = self._load_image(files[0])
        imageB, _ = self._load_image(files[1])
        mask, _ = self._load_image(str(files[2]))
        mask = np.clip(mask, a_min=0, a_max=1).astype(np.uint8)
        if not self.load_area_mask:
            return imageA, imageB, mask, meta
        else:
            area_mask, _ = self._load_image(str(files[3]))
            area_mask = np.clip(area_mask, a_min=0, a_max=1).astype(np.uint8)
            return imageA, imageB, mask, area_mask, meta

    def _load_tensor_index(self, index: int):
        area_mask = None
        if not self.load_area_mask:
            image1, image2, mask, meta = self._load_raw_index(index)
        else:
            image1, image2, mask, area_mask, meta = self._load_raw_index(index)
        image1_tensor = torch.from_numpy(image1)
        image2_tensor = torch.from_numpy(image2)
        mask_tensor = torch.from_numpy(mask).to(torch.uint8)
        raw_img_tensor = torch.cat([image1_tensor, image2_tensor])

        if area_mask is None:
            return raw_img_tensor.to(torch.float), mask_tensor, meta
        else:
            area_mask_tensor = torch.from_numpy(area_mask).to(torch.uint8)
            return raw_img_tensor.to(torch.float), mask_tensor, area_mask_tensor, meta

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
        if self.no_tile:
            total_tiles = len(self.file_list)
            index_map = {i: (i, i) for i in range(total_tiles)}

            for i in range(total_tiles):
                files = self.file_list[i]
                image1, _ = self._load_image(files[0])
                image1_tensor = torch.from_numpy(image1)
                raw_img_tensor = torch.cat([image1_tensor, image1_tensor])
                image_shapes[i] = list(raw_img_tensor.shape)
        else:
            for i in range(len(self.file_list)):
                files = self.file_list[i]
                image1, _ = self._load_image(files[0])
                if self.tile_mode != "constant":
                    if image1.shape[1] < self.tile_shape[0] or image1.shape[2] < self.tile_shape[1]:
                        continue
                image1_tensor = torch.from_numpy(image1)
                
                if self.index_no_mask:
                    raw_img_tensor = torch.cat([image1_tensor, image1_tensor])
                else:
                    mask, _ = self._load_image(files[2])
                    mask_tensor = torch.from_numpy(mask)
                    raw_img_tensor = torch.cat([image1_tensor, image1_tensor, mask_tensor])

                tiler = Tiler(data_shape=raw_img_tensor.shape,
                    tile_shape=(raw_img_tensor.shape[0], self.tile_shape[0], self.tile_shape[1]),
                    overlap=(raw_img_tensor.shape[0]-1, self.tile_overlap[0], self.tile_overlap[1]),
                    mode=self.tile_mode,
                    channel_dimension=0)

                image_shapes[i] = list(raw_img_tensor.shape)
                if not self.index_no_mask:
                    image_shapes[i][0] -= 1 # In this case we need to remove the mask channel from the shape
                tile_shape = tiler.get_mosaic_shape(with_channel_dim=True)
                num_tiles = tile_shape[0] * tile_shape[1] * tile_shape[2]

                if self.index_no_mask:
                    # Map chip index to file index
                    for j in range(total_tiles, total_tiles + num_tiles):
                        index_map[j] = (i, j - total_tiles)

                    total_tiles += num_tiles
                else:
                    # Skip files that don't have mask data
                    def is_in_aoi(mask_tile):
                        num_pixels_in_aoi = mask_tile.sum()
                        return num_pixels_in_aoi > 1 # At least 1 white GT mask pixel
                    
                    full_img_numpy = raw_img_tensor.cpu().numpy()
                    full_image_shape = image_shapes[i].copy()
                    full_image_shape[0] += 1 # Add mask channel to channel size

                    # When index_no_mask is False, we don't want to index any tiles
                    # that don't have sufficient white mask pixels
                    added = 0
                    for j in range(total_tiles, total_tiles + num_tiles):
                        chip_index = j - total_tiles
                        img_mask_tile = tiler.get_tile(full_img_numpy, chip_index, copy_data = False)
                        mask_tile = img_mask_tile[full_image_shape[0] - 1, :, :]
                        if is_in_aoi(mask_tile):
                            index_map[total_tiles + added] = (i, chip_index)
                            added += 1
                            
                    total_tiles += added

        return total_tiles, index_map, image_shapes

    def _load_image(self, path: string):
        """Load a single image.
        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as img:
            image = img.read()  # rasterio reads images as (bands, height, width)
            meta = img.meta
            if self.bands == 'all':
                image = image.astype(np.int16)
            return image, meta

    def _get_full_image_shape(self, file_index):
        full_image_shape = self.image_shapes_map[file_index].copy()
        mask_channels = 1 if not self.load_area_mask else 2
        full_image_shape[0] = full_image_shape[0] + mask_channels
        return full_image_shape

    def _get_tiler(self, file_index, channels_override = None):
        full_image_shape = self._get_full_image_shape(file_index)
        if self.no_tile:
            return None, full_image_shape

        if channels_override is not None and isinstance(channels_override, int):
            full_image_shape[0] = channels_override

        tiler = Tiler(data_shape=full_image_shape,
              tile_shape=(full_image_shape[0], self.tile_shape[0], self.tile_shape[1]),
              overlap=(full_image_shape[0]-1, self.tile_overlap[0], self.tile_overlap[1]),
              mode=self.tile_mode,
              channel_dimension=0)
        return tiler, full_image_shape

    def _get_file_index(self, chip_index):
        return self.chip_index_map[chip_index]

    def get_tile_file(self, chip_index):
        file_index, _ = self._get_file_index(chip_index)
        return self.file_list[file_index]

    def __getitem__(self, chip_index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            chip_index: index to return

        Returns:
            data and label at that index
        """

        (file_index, relative_chip_index) = self._get_file_index(chip_index)
        if not self.no_tile:
            tiler, full_image_shape = self._get_tiler(file_index)
            if not self.load_area_mask:
                img1, img2, mask, meta = self._load_raw_index(file_index)
            else:
                img1, img2, mask, area_mask, meta = self._load_raw_index(file_index)

            try:
                if not self.load_area_mask:
                    img_full = np.concatenate((img1, img2, mask))
                else:
                    img_full = np.concatenate((img1, img2, mask, area_mask))
            except ValueError as e:
                print(e)
                print(self.file_list[file_index])

            img_mask_tile = tiler.get_tile(img_full, relative_chip_index, copy_data = False)

            if not self.load_area_mask:
                img_tile, mask_tile = img_mask_tile[0:full_image_shape[0] - 1, :, :], \
                                          img_mask_tile[full_image_shape[0] - 1, :, :]
            else:
                img_tile, mask_tile, area_mask_tile = img_mask_tile[0:full_image_shape[0] - 2, :, :], \
                                                          img_mask_tile[full_image_shape[0] - 2, :, :], \
                                                          img_mask_tile[full_image_shape[0] - 1, :, :]
        else:
            if not self.load_area_mask:
                img1, img2, mask_tile, meta = self._load_raw_index(file_index)
            else:
                img1, img2, mask_tile, area_mask_tile, meta = self._load_raw_index(file_index)
                area_mask_tile = np.squeeze(area_mask_tile, 0)
            
            mask_tile = np.squeeze(mask_tile, 0)
            img_tile = np.concatenate((img1, img2))

        img_tile_tensor = torch.from_numpy(img_tile).to(torch.float)
        mask_tile_tensor = torch.from_numpy(mask_tile).to(torch.uint8).unsqueeze(0)
        
        if self.load_area_mask:
            area_mask_tile_tensor = torch.from_numpy(area_mask_tile).to(torch.uint8).unsqueeze(0)
            sample = {"image": img_tile_tensor, "mask": mask_tile_tensor, 'area_mask': area_mask_tile_tensor, 'meta': meta}
        else:
            sample = {"image": img_tile_tensor, "mask": mask_tile_tensor, 'meta': meta}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return self.total_dataset_length

    def get_tile_relative_offset(self, tile_idx):
        if not self.no_tile:
            (file_index, relative_chip_index) = self._get_file_index(tile_idx)
            tiler, _ = self._get_tiler(file_index)

            bbox = tiler.get_tile_bbox(relative_chip_index)
            TL = bbox[0][::-1]
            # BR = bbox[1][::-1]
            return TL[0], TL[1] #x, y in standard image coordinates
        else:
            return 0, 0

    def get_tile_metadata(self, tile_idx, mask = False, no_tile = None):
        """
          Get tile-relative metadata. This is pulled from the metadata of the
          original full-sized GeoTIFF. If tiles are enabled, the geolocation
          metadata is transformed to be accurate for the specific tile in
          question.

          Args:
            - tile_idx: The index of the tile.
            - mask: (Default = False) If True, it will return the metadata for
                the tile's mask layer.
            - no_tile: (Default = None) If True, returns the metadata for the
                full-sized original image. If False, only returns the metadata
                for the given tile_idx. If None, uses the dataset's no_tile setting.
          Returns:
            The tile-relative metadata. The original metadata is not modified.
        """
        if no_tile is None:
            no_tile = self.no_tile

        img_dict = self[tile_idx]
        meta = img_dict['meta'].copy()
        if mask:
            meta['count'] = 1

        if not no_tile:
            x,y = self.get_tile_relative_offset(998)
            meta['transform'] = meta['transform'] * Affine.translation(x, y)
            meta['width'] = img_dict['image'].shape[1]
            meta['height'] = img_dict['image'].shape[2]
            return meta
        else:
            return meta

    def get_image_tile_ranges(self):
        tile_idx_map = {}
        for chip_index in self.chip_index_map.keys():
            file_idx, rel_tile_idx = self.chip_index_map[chip_index]
            if file_idx in tile_idx_map:
                tile_idx_map[file_idx].append(chip_index)
            else:
                tile_idx_map[file_idx] = [chip_index]

        for i in tile_idx_map.keys():
            max_idx = max(tile_idx_map[i])
            min_idx = min(tile_idx_map[i])
            tile_idx_map[i] = (min_idx, max_idx)

        return tile_idx_map
    
    def get_facilities(self):
        """
        Get a full list of facilities in the dataset.
        """
        return list(self.facilities_list)

    def get_facility_tile_indices(self, facility):
        """
        For the given facillity name, get all the dataset tile indices for that facility.
        """
        indices = []
        for chip_idx in range(len(self)):
            file_info = self.get_tile_file(chip_idx)
            if facility == file_info[-4]:
                indices.append(chip_idx)

        return indices

    def get_facility_file_indices(self, facility):
        """
        For the given facility name, get all the dataset file indices for that facility.
        """
        indices = []
        valid_indices = set(self.chip_index_map.keys())
        for file_index in range(len(self.file_list)):
            if facility == self.file_list[file_index][-4] and file_index in valid_indices:
                indices.append(file_index)
        return indices
    
    def get_file_info(self, file_idx):
        return self.file_list[file_idx]

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get_normalization_values(self):
        return self.__class__.GetNormalizationValues(self.bands)

    def split_images(self, images):
        band_nums = {
            'rgb': 3,
            'all': 13,
            'rgbnir': 4
        }
        n_bands = band_nums[self.bands]
        pre, post = images[:, 0:n_bands], images[:, n_bands:2*n_bands]
        return pre, post

    # Adapted from https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/oscd.html#OSCD
    def plot(
        self,
        sample,
        show_titles: bool = True,
        suptitle = None,
        alpha: float = 0.5,
    ):
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2

        inds_map = {
            'rgb': [0, 1, 2],
            'rgbnir': [0, 1, 2],
            'all': [3, 2, 1]
        }

        rgb_inds = inds_map[self.bands]

        def get_masked(img) -> "np.typing.NDArray[np.uint8]":
            rgb_img = img[rgb_inds].float().numpy()
            per02 = np.percentile(rgb_img, 2)
            per98 = np.percentile(rgb_img, 98)
            rgb_img = (np.clip((rgb_img - per02) / (per98 - per02), 0, 1) * 255).astype(
                np.uint8
            )
            array: "np.typing.NDArray[np.uint8]" = draw_semantic_segmentation_masks(
                torch.from_numpy(rgb_img),
                sample["mask"],
                alpha=alpha,
                colors=self.colormap,
            )
            return array

        idx = sample["image"].shape[0] // 2
        image1 = get_masked(sample["image"][:idx])
        image2 = get_masked(sample["image"][idx:])
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Pre change")
            axs[1].set_title("Post change")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    @classmethod
    def GetNormalizationValues(cls, bands="rgb"):
        return cls.normalisation_map[bands]

    @classmethod
    def GetNormalizeTransform(cls, bands='rgb'):
        # Mean/STD as 3 or 13 channels/bands
        [mean, std] = cls.GetNormalizationValues(bands)
        # We are loading our dataset as a stack of two images (pre/post) as 2 *
        # the number of channels in one image. So for RGB, our tensor has 6
        # channels instead of 2. So we need to stack our normalisation tensor
        # so it matches the image tensor.
        mean, std = torch.cat((mean, mean), dim=0), torch.cat((std, std), dim=0)
        return NormalizeImageDict(mean=mean, std=std)

    @classmethod
    def CalcMeanVar(cls, root, split="train", bands="rgb"):
        def t(img_dict):
            return {'image': img_dict['image'].to(torch.float), 'mask': img_dict['mask']}

        dataset = cls(root=root, split=split, bands=bands, transforms = None, tile_size = None)

        def preproc_img(img_dict):
            images = img_dict['image']
            images = images.unsqueeze(0)
            batch_samples = images.size(0)
            B, C, W, H = images.size()
            # Separate the tensor into two tensors of shape (B, 3, W, H)
            image1 = images[:, :(C//2), :, :]
            image2 = images[:, (C//2):, :, :]
            # Stack them to get a tensor of shape (2B, 3, W, H)
            images = torch.cat((image1, image2), dim=0)
            images = images.view(-1, C//2, W, H)
            return images

        def compute_dataset_mean_std(dataset):
            ex_img = preproc_img(dataset[0]).shape[1]
            total_sum = torch.zeros(ex_img)
            total_sq_sum = torch.zeros(ex_img)
            total_num_pixels = 0

            for i in range(len(dataset)):
                image = preproc_img(dataset[0]).float()
                total_sum += image.sum(dim=[0, 2, 3])  # sum of pixel values in each channel
                total_sq_sum += (image ** 2).sum(dim=[0, 2, 3])  # sum of squared pixel values in each channel
                total_num_pixels += image.shape[0] * image.shape[2] * image.shape[3]  # total number of pixels in an image

            mean = total_sum / total_num_pixels  # mean = total sum / total number of pixels
            std = (total_sq_sum / total_num_pixels - mean ** 2) ** 0.5  # std = sqrt(E[X^2] - E[X]^2)

            return mean, std
        return compute_dataset_mean_std(dataset)


class OMS2CDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the OMS2CD dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.
    """
    mean = OMS2CD.mean
    std = OMS2CD.std

    def __init__(
        self,
        batch_size: int = 16,
        bands: str = 'rgb',
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new OSCDDataModule instance. 

        Args:
            batch_size: Size of each mini-batch.
            # patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            #     Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NonGeoDataModule`.

            Example:
                datamodule = OMS2CDDataModule(root='OMS2CD', bands='rgb', load_area_mask=False,
                              batch_size=16, tile_mode="constant", index_no_mask=True, stride=100)
        """
        super().__init__(OMS2CD, batch_size, num_workers, **kwargs)
        self.mean, self.std = OMS2CD.GetNormalizationValues(bands)

        # Change detection, 2 images from different times
        self.mean = torch.cat((self.mean, self.mean), dim=0)
        self.std = torch.cat((self.std, self.std), dim=0)

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
        )
        self.transforms = {stage: None for stage in ['fit', 'validate', 'test', 'predict']}
        def collate_fn(batch):
            collated_batch = {}

            # Iterate over each key in the batch
            for key in batch[0].keys():
                # If the data under this key is a tensor, stack it
                if isinstance(batch[0][key], torch.Tensor):
                    collated_batch[key] = torch.stack([item[key] for item in batch])
                # If the data under this key is not a tensor, just store it as a list
                else:
                    collated_batch[key] = [item[key] for item in batch]
                    
            return collated_batch
            
        self.collate_fn = collate_fn

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.train_dataset = OMS2CD(split="train", **self.kwargs)
            self.train_dataset.set_transforms(self.transforms['fit'])
            self.val_dataset = OMS2CD(split="val", **self.kwargs)
            self.val_dataset.set_transforms(self.transforms['validate'])
        if stage in ["test"]:
            self.test_dataset = OMS2CD(split="test", **self.kwargs)
            self.test_dataset.set_transforms(self.transforms['test'])
    
    def set_transforms(self, transforms: Any, stage: str = "fit"):
        assert stage in ['fit', 'validate', 'test', 'predict']
        self.transforms[stage] = transforms
