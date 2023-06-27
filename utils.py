import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional
from torchvision.transforms.functional import to_pil_image
from torchgeo.datasets.utils import draw_semantic_segmentation_masks
import numpy as np
import kornia.augmentation as K

def plot_prediction(
    model: torch.nn.Module,
    sample,
    bands: str,
    colormap,
    alpha: float = 0.4,
    show_titles: bool = True,
    suptitle: Optional[str] = None,
) -> None:
    model = model.eval()
    with torch.no_grad():
        img = sample['image']
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        mask_pred = model(img).squeeze(1).cpu()

    if len(img.shape) > 3:
        img = img.squeeze(0)
    mask_pred = (mask_pred > 0.01).float()

    rgb_inds = [3, 2, 1] if bands == "all" else [0, 1, 2]

    def get_masked(img: torch.Tensor, mask: torch.Tensor) -> "np.typing.NDArray[np.uint8]":
        rgb_img = img[rgb_inds].float().numpy()
        per02 = np.percentile(rgb_img, 2)
        per98 = np.percentile(rgb_img, 98)
        rgb_img = (np.clip((rgb_img - per02) / (per98 - per02), 0, 1) * 255).astype(
            np.uint8
        )
        array: "np.typing.NDArray[np.uint8]" = draw_semantic_segmentation_masks(
            torch.from_numpy(rgb_img),
            mask,
            alpha=alpha,
            colors=colormap,
        )
        return array

    idx = img.shape[0] // 2
    image1 = get_masked(img[:idx], sample["mask"])
    image2 = get_masked(img[idx:], sample["mask"])
    image3 = get_masked(img[:idx], mask_pred)
    image4 = get_masked(img[idx:], mask_pred)
    image5 = to_pil_image(mask_pred.byte())
    image6 = to_pil_image(sample["mask"].byte())

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    axs[0, 0].imshow(image1)
    axs[0, 0].axis("off")
    axs[0, 1].imshow(image2)
    axs[0, 1].axis("off")
    axs[1, 0].imshow(image3)
    axs[1, 0].axis("off")
    axs[1, 1].imshow(image4)
    axs[1, 1].axis("off")
    axs[2, 0].imshow(image5)
    axs[2, 0].axis("off")
    axs[2, 1].imshow(image6)
    axs[2, 1].axis("off")

    if show_titles:
        axs[0, 0].set_title("Pre change")
        axs[0, 1].set_title("Post change")
        axs[1, 0].set_title("Pre change with predicted mask")
        axs[1, 1].set_title("Post change with predicted mask")
        axs[2, 0].set_title("Predicted mask")
        axs[2, 1].set_title("Ground truth mask")

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.show()

def crop_sample(dataset, index: int, size: int = 256):
    from torchvision.transforms import CenterCrop

    sample = dataset[index]
    cropper = CenterCrop(size)

    sample['image'] = cropper(sample['image'])
    sample['mask'] = cropper(sample['mask'])

    return sample

def normalize_sample(sample, mean, std):
    image = sample['image'].float()
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    normalize = K.Normalize(mean, std)
    normalized_image = normalize(image)
    sample['image'] = normalized_image
    return sample

def get_oscd_norm_coefficients(bands="rgb"):
    # mean = OSCDDataModule.mean
    # std = OSCDDataModule.std
    mean = torch.tensor([1571.1372, 1365.5087, 1284.8223, 1298.9539, 1431.2260, 1860.9531,
                    2081.9634, 1994.7665, 2214.5986,  641.4485,   14.3672, 1957.3165,
                    1419.6107])
    std =  torch.tensor([274.9591,  414.6901,  537.6539,  765.5303,  724.2261,  760.2133,
                    848.7888,  866.8081,  920.1696,  322.1572,    8.6878, 1019.1249,
                    872.1970])
    if bands == "rgb":
        mean = mean[[3, 2, 1]]
        std = std[[3, 2, 1]]

    mean = torch.cat([mean, mean], dim=0)
    std = torch.cat([std, std], dim=0)
    return mean, std

def get_omcd_norm_coefficients():
    # mean = OSCDDataModule.mean
    # std = OSCDDataModule.std
    mean = torch.tensor([121.7963, 123.6833, 116.5527])
    std = torch.tensor([67.2949, 68.0268, 65.1758])

    mean = torch.cat([mean, mean], dim=0)
    std = torch.cat([std, std], dim=0)
    return mean, std