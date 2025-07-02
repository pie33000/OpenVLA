import torch
import torch.nn as nn
from transformers import SiglipVisionModel  


class VisionEncoder(nn.Module):
    def __init__(self, dino_model_name: str = "dinov2_vits14", siglip_model_name: str = "google/siglip-so400m-patch14-224", dim: int = 1024):
        super(VisionEncoder, self).__init__()
        self.dim = dim
        self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_model_name)
        self.siglip_model = SiglipVisionModel.from_pretrained(
                 "google/siglip-so400m-patch14-224"           # downloads only vision weights
               )
        self.proj = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.GELU(),
            nn.Linear(1024, self.dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        dino_out = self.dino_model(x)
        siglip_out = self.siglip_model(x)
        siglip_out = siglip_out.pooler_output
        out = torch.cat([dino_out, siglip_out], dim=-1)
        out = self.proj(out)
        out = out.view(B, 1, -1)
        return out