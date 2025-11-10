from torchvision import transforms
import torch
import numpy as np

class PathMNISTPreprocessor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def preprocess(self, pil_image):
        return self.transform(pil_image)

    def tensor_to_numpy(self, batch_tensor):
        """
        Convert a normalized tensor batch (1x3x224x224) into a [0,1] HxWxC numpy image.
        Detaches from autograd before converting to numpy.
        """
        t = batch_tensor.detach().squeeze(0).cpu()  # detach to avoid RuntimeError
        denorm = t.clone()
        for c, (m, s) in enumerate(zip(self.mean, self.std)):
            denorm[c] = denorm[c] * s + m
        img = denorm.permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)
