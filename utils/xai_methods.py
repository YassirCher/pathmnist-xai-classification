import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np
from matplotlib.figure import Figure
from captum.attr import IntegratedGradients, LayerGradCam
import io, base64

class XAIExplainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(model)
        self.layer = model.resnet.layer4[-1]
        self.layer_gc = LayerGradCam(model, self.layer)

    def gradcam(self, x, target):
        x.requires_grad = True
        attr = self.layer_gc.attribute(x, target=target)
        # Use bilinear interpolation for smoother upsampling
        up = LayerGradCam.interpolate(attr, (224, 224), interpolate_mode='bilinear')
        a = up.squeeze().detach().cpu().numpy()
        if a.ndim == 3:
            a = a.mean(axis=-1)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        return a

    def integrated_gradients(self, x, target):
        x.requires_grad = True
        baseline = torch.zeros_like(x)
        # Increase n_steps for smoother gradients
        attr = self.ig.attribute(x, baselines=baseline, target=target, n_steps=100)
        
        # Take absolute values and sum across color channels
        a = attr.squeeze().abs().sum(dim=0).detach().cpu().numpy()
        
        # Apply Gaussian smoothing to reduce noise and create smoother visualization
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, sigma=2.0)  # Smooth with Gaussian filter
        
        # Normalize to [0, 1]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        
        return a

    @staticmethod
    def overlay(original_np, heatmap, alpha=0.5):
        # Ensure heatmap is 2D normalized
        hm = np.asarray(heatmap)
        if hm.ndim == 3:
            hm = hm.mean(axis=-1)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        
        # Apply slight Gaussian smoothing to heatmap for smoother overlay
        from scipy.ndimage import gaussian_filter
        hm = gaussian_filter(hm, sigma=0.5)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        
        import matplotlib.pyplot as plt  # safe with Agg
        cmap = plt.cm.jet(hm)[..., :3]
        out = original_np * (1 - alpha) + cmap * alpha
        return np.clip(out, 0, 1)

    @staticmethod
    def to_base64(np_img):
        """
        Convert numpy image to base64 with high quality and smooth interpolation
        """
        fig = Figure(figsize=(4, 4), dpi=150)  # Higher DPI for better quality
        ax = fig.add_subplot(111)
        ax.imshow(np_img, interpolation='bilinear')  # Smooth interpolation instead of blocky
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
