import os
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import NonGeoDataset
import numpy as np
from torchgeo.datasets.utils import draw_semantic_segmentation_masks
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class OMCD(NonGeoDataset):
    """OMCD Dataset
    As described in:
    J. Li, J. Xing, S. Du, S. Du, C. Zhang, and W. Li, “Change Detection of Open-
    Pit Mine Based on Siamese Multiscale Network,” IEEE Geoscience and Remote 
    Sensing Letters, vol. 20, pp. 1-5, 2023, doi: 10.1109/LGRS.2022.3232763.

    Hosted: https://figshare.com/s/ae4e8c808b67543d41e9
    Download URL: https://figshare.com/ndownloader/files/36683622?private_link=ae4e8c808b67543d41e9

    License: CC By 4.0 - https://creativecommons.org/licenses/by/4.0/
    """

    colormap = ["blue"]
    def __init__(self, root, split="train", transforms = None):
        if split == "train":
            subset_path = "training dataset(data augmentation)"
        elif split == "test":
            subset_path = "testing dataset(cropping)"
        elif split == "test_no_crop":
            subset_path = "testing dataset(no cropping)"
        elif split == "train_no_augment":
            subset_path = "training dataset(no data augmentation)"
        else:
            raise ValueError("Invalid dataset split.")

        self.root_dir = os.path.join(root, subset_path)
        self.file_list = self._build_index()  # using build_index method to build the file list
        self.transforms = transforms

    def _build_index(self):
        self.imageA_dir = os.path.join(self.root_dir, 'imageA')
        self.imageB_dir = os.path.join(self.root_dir, 'imageB')
        self.masks_dir = os.path.join(self.root_dir, 'mask')
        index_list = []
        for filename in os.listdir(self.imageA_dir):
            # construct file paths for imageA and imageB
            imageA_path = os.path.join(self.imageA_dir, filename)
            imageB_path = os.path.join(self.imageB_dir, filename)

            # construct mask path, assuming mask files are named 'label*.tif'
            mask_filename = filename.replace('image', 'label')
            mask_path = os.path.join(self.masks_dir, mask_filename)
            if os.path.isfile(imageA_path) and os.path.isfile(imageB_path) and os.path.isfile(mask_path):
                index_list.append((imageA_path, imageB_path, mask_path))

        return index_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        imageA_path, imageB_path, mask_path = self.file_list[index]

        with rasterio.open(imageA_path) as img:
            imageA = img.read()  # rasterio reads images as (bands, height, width)
        with rasterio.open(imageB_path) as img:
            imageB = img.read()
        with rasterio.open(mask_path) as img:
            mask = img.read()

        image = torch.from_numpy(np.concatenate((imageA, imageB), axis=0).astype(np.uint8)).to(torch.float)  # stack along the channel axis
        mask = torch.from_numpy(mask).to(torch.uint8)
        sample = {'image': image, 'mask': mask}
        
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

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

        rgb_inds = [0, 1, 2]

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

    @staticmethod
    def CalcMeanVar(root, split="train"):
        def t(img_dict):
            return {'image': img_dict['image'].to(torch.float), 'mask': img_dict['mask']}

        dataset = OMCD(root=root, split=split, transforms = t)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        def preproc_img(img_dict):
            images = img_dict['image']
            batch_samples = images.size(0)
            B, C, W, H = images.size()
            # Separate the tensor into two tensors of shape (B, 3, W, H)
            image1 = images[:, :(C//2), :, :]
            image2 = images[:, (C//2):, :, :]
            # Stack them to get a tensor of shape (2B, 3, W, H)
            images = torch.cat((image1, image2), dim=0)
            images = images.view(-1, C//2, W, H)
            return images

        def compute_dataset_mean_std(dataloader):
            ex_img = preproc_img(next(iter(dataloader))).shape[1]
            total_sum = torch.zeros(ex_img)
            total_sq_sum = torch.zeros(ex_img)
            total_num_pixels = 0

            for batch in dataloader:
                image = preproc_img(batch).float()
                total_sum += image.sum(dim=[0, 2, 3])  # sum of pixel values in each channel
                total_sq_sum += (image ** 2).sum(dim=[0, 2, 3])  # sum of squared pixel values in each channel
                total_num_pixels += image.shape[0] * image.shape[2] * image.shape[3]  # total number of pixels in an image

            mean = total_sum / total_num_pixels  # mean = total sum / total number of pixels
            std = (total_sq_sum / total_num_pixels - mean ** 2) ** 0.5  # std = sqrt(E[X^2] - E[X]^2)

            return mean, std
        return compute_dataset_mean_std(loader)