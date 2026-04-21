import torch
import torch.nn as nn

class Dinov3Backbone(nn.Module):
    def __init__(self, backbone_model, embed_dim=384, out_channels=256):
        super().__init__()
        self.backbone = backbone_model
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = 16  # ViT-S/16 的 patch size

        # Transform ViT features to out_channels (stride 16)
        self.conv_c4 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        
        # FPN layers
        # Stride 8 (Deconv from stride 16)
        self.deconv_c3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # Stride 32 (MaxPool from stride 16)
        self.pool_c5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Stride 64 (MaxPool from stride 32)
        self.pool_c6 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        B, C, H, W = x.shape
        # 计算特征图高宽
        H_feat = H // self.patch_size
        W_feat = W // self.patch_size

        features_dict = self.backbone.forward_features(x)
        # 提取 patch tokens
        if 'x_norm_patchtokens' in features_dict:
            patch_tokens = features_dict['x_norm_patchtokens']  # [B, N, D]
        elif 'x_patchtokens' in features_dict:
            patch_tokens = features_dict['x_patchtokens']
        else:
            patch_tokens = list(features_dict.values())[0]

        B, N, D = patch_tokens.shape
        expected_N = H_feat * W_feat

        # 如果包含 CLS token，去掉它
        if N == expected_N + 1:
            patch_tokens = patch_tokens[:, 1:, :]
        elif N != expected_N:
            raise ValueError(
                f"Patch token count mismatch: expected {expected_N} (from image size {H}x{W}), got {N}. "
                f"Make sure image dimensions are divisible by patch size ({self.patch_size})."
            )

        # 重塑为特征图 [B, D, H_feat, W_feat]
        feat_map = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, D, H_feat, W_feat)
        
        # Base feature map (Stride 16)
        c4 = self.conv_c4(feat_map)
        
        # Stride 8
        c3 = self.deconv_c3(c4)
        
        # Stride 32
        c5 = self.pool_c5(c4)
        
        # Stride 64
        c6 = self.pool_c6(c5)
        
        return {
            '0': c3,
            '1': c4,
            '2': c5,
            '3': c6
        }