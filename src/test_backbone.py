import torch
from dinov3_backbone import Dinov3Backbone
from pathlib import Path


REPO_DIR = "/Users/mises/Python/dinov3"
WEIGHTS_PATH = "/Users/mises/Python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# 加载模型
backbone_model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH)
backbone_model.eval()

# 包装 backbone
dinov3_backbone = Dinov3Backbone(backbone_model, embed_dim=384, out_channels=256)

# 测试
dummy_input = torch.randn(1, 3, 256, 256)
output = dinov3_backbone(dummy_input)
print("Output keys:", output.keys())
print("Feature map shape:", output['0'].shape)