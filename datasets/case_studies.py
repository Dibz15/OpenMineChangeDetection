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


from . import OMS2CD
import csv
import os, re
import torch
from .transforms import NormalizeImageDict
import numpy as np

class CaseStudyDataset(OMS2CD):
    def __init__(self, root, stride = 128):
        super().__init__(
            root,
            "train",
            "rgb",
            None,
            tile_mode="constant",
            stride = stride,
            tile_size = 256,
            load_area_mask = False,
            index_no_mask = True
        )

    def _create_mapping(self):
        """
            Creates a mapping file for Sentinel-2 image pairs in the dataset.

            The mapping file contains the following columns:
            - id: Unique identifier for each pair of images.
            - imageA: Filename of the first image (pre-change).
            - imageB: Filename of the second image (post-change).
            - facility: Name of the facility corresponding to the image pair.
            - predate: Date of the first image (pre-change).
            - postdate: Date of the second image (post-change).

            This function scans the root directory of the dataset and identifies pairs of Sentinel-2 images 
            that belong to the same facility. It then writes the corresponding file names and facility 
            information into the mapping file as bi-temporal pairs for training/inference.

            Note: If the mapping file exists, this function doesn't do anything.

            Returns:
            None
        """
        source_dir = self.root_dir
        mapping_file_path = os.path.join(source_dir, 'mapping.csv')
        if os.path.isfile(mapping_file_path):
            return
        else:
            print(f'Data mapping file being created at {mapping_file_path}.')
        # Get the list of s2 files and sort them
        s2_files = [f for f in os.listdir(source_dir) if f.startswith('s2_') and f.endswith('.tif')]
        s2_files.sort()

        # Initialize the facility name and date
        prev_facility = None
        prev_date = None
        counter = 1
        imageAWrite = True

        # Open the mapping file in write mode
        with open(mapping_file_path, 'w', newline='') as mapping_file:
            writer = csv.writer(mapping_file)
            writer.writerow(['id', 'imageA', 'imageB', 'facility', 'predate', 'postdate'])  # Write the header

            for s2_file in s2_files:
                # Parse the facility name and date from the filename
                s2_file_without_ext = s2_file.replace('.tif', '')
                s2_prefix_and_rest, date_str = s2_file_without_ext.rsplit('_', 1)
                s2_prefix, rest = s2_prefix_and_rest.split('_', 1)
                facility = re.split(r'[\d-]', rest)[0].strip('_')
                postdate = date_str

                if facility == prev_facility and not imageAWrite:
                    # If the facility is the same as the previous one, write the file to imageB
                    new_filename = s2_file
                    writer.writerow([str(counter).zfill(4), prev_filename, new_filename, facility, prev_date, postdate])
                    print(f'Wrote {counter:04d} {prev_filename}, {new_filename}, {facility} to mapping file')
                    imageAWrite = True
                    counter += 1

                # Write the s2 file to imageA
                if imageAWrite:
                    imageAWrite = False

                # Update the facility name and date
                prev_filename = s2_file
                prev_facility = facility
                prev_date = postdate

    def _build_index(self):
        """
            Builds an index list of image pairs and corresponding facility information.

            The index list contains tuples for each image pair, with the following elements:
            - imageA_path: File path of the first image (pre-change).
            - imageB_path: File path of the second image (post-change).
            - facility: Name of the facility corresponding to the image pair.
            - predate: Date of the first image (pre-change).
            - postdate: Date of the second image (post-change).
            - id: Unique identifier for each pair of images.

            This function relies on the existence of the mapping file (mapping.csv) in the dataset's root directory,
            so it first runs _create_mapping()

            Returns:
            index_list (list): A list of tuples containing image pair information and facility names.
        """
        self._create_mapping()
        index_list = []
        facilities = set()
        with open(os.path.join(self.root_dir, 'mapping.csv'), 'r', newline='') as mapping_file:
            reader = csv.DictReader(mapping_file)
            for row in reader:
                id = row['id']
                facility = row['facility']
                imageA_path = os.path.join(self.root_dir, row['imageA'])
                imageB_path = os.path.join(self.root_dir, row['imageB'])
                if os.path.isfile(imageA_path) and os.path.isfile(imageB_path):
                    index_list.append((imageA_path, imageB_path, facility, row['predate'], row['postdate'], id))
                    facilities.add(facility)

        self.facilities_list = facilities
        return index_list

    def _load_raw_index(self, index: int):
        files = self.file_list[index]
        imageA, meta = self._load_image(files[0])
        imageB, _ = self._load_image(files[1])
        if not self.load_area_mask:
            return imageA, imageB, meta, files[2]

    def _load_tensor_index(self, index: int):
        image1, image2, meta, facility = self._load_raw_index(index)
        image1_tensor = torch.from_numpy(image1)
        image2_tensor = torch.from_numpy(image2)
        raw_img_tensor = torch.cat([image1_tensor, image2_tensor])
        return raw_img_tensor.float(), meta, facility

    def _get_full_image_shape(self, file_index):
        full_image_shape = self.image_shapes_map[file_index].copy()
        return full_image_shape

    def __getitem__(self, chip_index: int):
        """Return an index within the dataset.

        Args:
            chip_index: index to return

        Returns:
            data and label at that index
        """

        (file_index, relative_chip_index) = self._get_file_index(chip_index)
        if not self.no_tile:
            tiler, _ = self._get_tiler(file_index)
            img1, img2, meta, facility = self._load_raw_index(file_index)

            try:
                img_full = np.concatenate((img1, img2))
            except ValueError as e:
                print(e)
                print(self.file_list[file_index])

            img_tile = tiler.get_tile(img_full, relative_chip_index, copy_data = False)
        else:
            img1, img2, meta, facility = self._load_raw_index(file_index)
            img_tile = np.concatenate((img1, img2))

        img_tile_tensor = torch.from_numpy(img_tile).to(torch.float)
        sample = {"image": img_tile_tensor, 'meta': meta, 'facility': facility}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_facility_tile_indices(self, facility):
        """
        For the given facillity name, get all the dataset tile indices for that facility.
        """
        indices = []
        for chip_idx in range(len(self)):
            file_info = self.get_tile_file(chip_idx)
            if facility == file_info[2]:
                indices.append(chip_idx)

        return indices

    def get_facility_file_indices(self, facility):
        """
        For the given facility name, get all the dataset file indices for that facility.
        """
        indices = []
        valid_indices = set(self.chip_index_map.keys())
        for file_index in range(len(self.file_list)):
            if facility == self.file_list[file_index][2] and file_index in valid_indices:
                indices.append(file_index)
        return indices

    def get_facilities(self):
        """
        Get a full list of facilities in the dataset.
        """
        return list(self.facilities_list)

    @classmethod
    def GetNormalizeTransform(cls, bands='rgb', device='cpu'):
        # Mean/STD as 3 or 13 channels/bands
        [mean, std] = cls.GetNormalizationValues(bands)
        # We are loading our dataset as a stack of two images (pre/post) as 2 *
        # the number of channels in one image. So for RGB, our tensor has 6
        # channels instead of 2. So we need to stack our normalisation tensor
        # so it matches the image tensor.
        mean, std = torch.cat((mean, mean), dim=0, device=device), torch.cat((std, std), dim=0, device=device)
        return NormalizeImageDict(mean=mean, std=std)