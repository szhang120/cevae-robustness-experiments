import torch
import pyro
from visual_cevae import VisualCEVAE

class CEVAEWithZ(VisualCEVAE):
    """
    Extends the visual CEVAE (VisualCEVAE) by appending Z to X during training and inference.
    """
    def fit(self, x, t, y, z, **kwargs):
        x_extended = torch.cat([x, z], dim=1)
        return super().fit(x_extended, t, y, **kwargs)

    def ite(self, x, z):
        x_extended = torch.cat([x, z], dim=1)
        return super().ite(x_extended)
