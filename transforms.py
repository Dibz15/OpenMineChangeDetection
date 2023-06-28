from torchvision.transforms import Normalize

class NormalizeScale:
    def __init__(self, scale_factor=10000):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, sample):
        sample['image'] = sample['image'].to(torch.float) / self.scale_factor
        return sample

class NormalizeImageDict(Normalize):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def __call__(self, sample):
        sample['image'] = super().__call__(sample['image'])
        return sample