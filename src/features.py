"""
Turns a photo into a grid of patch feature vectors, using a frozen,
ImageNet-pretrained ResNet18 as a fixed feature extractor. No training
happens here -- the network's weights never change. We combine layer2
(finer texture detail) and layer3 (more abstract shape/pattern) so each
patch vector captures both, the same multi-scale choice PatchCore uses.
ResNet18 (rather than a larger backbone like WideResNet50) is a deliberate
speed tradeoff for CPU-only inference -- a few percent less peak accuracy
in exchange for roughly an order of magnitude faster feature extraction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import numpy as np

_device = torch.device("cpu")
torch.set_num_threads(4)

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_hook_outputs: dict = {}


def _hook(name):
    def _fn(module, inp, out):
        _hook_outputs[name] = out
    return _fn


def _load_model():
    global _model
    if _model is None:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(_device)
        model.layer2.register_forward_hook(_hook("layer2"))
        model.layer3.register_forward_hook(_hook("layer3"))
        _model = model
    return _model


@torch.no_grad()
def extract_patch_features(image: Image.Image) -> np.ndarray:
    """
    Returns a (grid_h, grid_w, feature_dim) array -- one feature vector per
    spatial patch of the image. grid_h/grid_w are ~28x28 for a 224x224 input;
    feature_dim is 128 (layer2) + 256 (layer3) = 384.
    """
    model = _load_model()
    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(_device)
    model(tensor)

    f2 = _hook_outputs["layer2"]  # [1, 128, 28, 28]
    f3 = _hook_outputs["layer3"]  # [1, 256, 14, 14]
    f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    combined = torch.cat([f2, f3_up], dim=1)  # [1, 384, 28, 28]

    combined = combined.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [28, 28, 1536]
    return combined
