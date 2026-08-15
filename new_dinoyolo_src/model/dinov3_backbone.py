import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    """Conv(/Deconv) + GroupNorm + SiLU。

    用 GroupNorm 而非 BatchNorm：检测任务 batch 通常较小（本项目为 8），
    GN 的统计量与 batch 大小无关，比 BN 稳定。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, transpose=False):
        super().__init__()
        if transpose:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=False)
        else:
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False
            )
        self.block = nn.Sequential(
            conv,
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Dinov3Backbone(nn.Module):
    """把 DinoV3 的 token 输出适配成多尺度特征图字典。

    **两种模式：**

    - ``out_indices=None``（默认，向后兼容）：只取 ViT 最后一层，stride 16 卷积成
      C4，再由 C4 反卷积出 stride 8、池化出 stride 32/64。三个尺度同源，且
      stride 32/64 分支无可学习参数。
    - ``out_indices=(a, b, c)``：真正的多尺度。从 ViT 的三个不同深度分别取
      patch token，各自投影成 P3/P4/P5，每级都带独立的投影 + 细化卷积。
      浅层保留局部纹理，深层承载语义，跨尺度融合才有实际信息可融。

    **输入输出：**
    - 输入: [B, 3, H, W]，H/W 需能被 patch_size 整除。
    - 输出: 字典，键 '0'/'1'/'2'/'3' 对应 stride 8/16/32/64。

    注意 ``get_intermediate_layers`` 按 block 索引升序返回，因此 ``out_indices``
    会在构造时排序，保证 shallow→deep 稳定映射到 P3→P5。
    """

    def __init__(self, backbone_model, embed_dim=384, out_channels=256, out_indices=None):
        super().__init__()
        self.backbone = backbone_model
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        # 从模型读取而非硬编码，避免换 patch size 的骨干时静默算错特征图尺寸。
        self.patch_size = getattr(backbone_model, "patch_size", 16)

        if out_indices is None:
            self.out_indices = None
            self.conv_c4 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
            self.deconv_c3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            self.pool_c5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool_c6 = nn.MaxPool2d(kernel_size=2, stride=2)
            return

        indices = sorted(set(int(index) for index in out_indices))
        if len(indices) != 4:
            raise ValueError(
                f"out_indices must contain exactly 4 distinct block indices, got {out_indices}"
            )
        num_blocks = len(backbone_model.blocks)
        if not all(0 <= index < num_blocks for index in indices):
            raise ValueError(
                f"out_indices {indices} out of range for a backbone with {num_blocks} blocks"
            )
        self.out_indices = tuple(indices)

        # 每级都从各自的 ViT 层独立投影（embed_dim -> out_channels），再各自细化。
        self.p2_proj = ConvGNAct(embed_dim, 128, kernel_size=4, stride=4, transpose=True)
        self.p3_proj = ConvGNAct(embed_dim, out_channels, kernel_size=2, stride=2, transpose=True)
        self.p4_proj = ConvGNAct(embed_dim, out_channels, kernel_size=1)
        self.p5_proj = ConvGNAct(embed_dim, out_channels, kernel_size=3, stride=2)
        self.p2_refine = ConvGNAct(128, 128, kernel_size=3)
        self.p3_refine = ConvGNAct(out_channels, out_channels, kernel_size=3)
        self.p4_refine = ConvGNAct(out_channels, out_channels, kernel_size=3)
        self.p5_refine = ConvGNAct(out_channels, out_channels, kernel_size=3)
        self.pool_c6 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.out_indices is None:
            return self._forward_single_source(x)
        return self._forward_multi_scale(x)

    def _forward_multi_scale(self, x):
        # reshape=True 直接返回 [B, D, H/patch, W/patch]，storage/cls token 已剥离。
        # 一次前向跑完全部 block，取三个中间结果，无额外计算开销。
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=True,
        )
        p2_tokens, shallow, middle, deep = features

        c2 = self.p2_refine(self.p2_proj(p2_tokens))  # stride 4
        c3 = self.p3_refine(self.p3_proj(shallow))    # stride 8
        c4 = self.p4_refine(self.p4_proj(middle))     # stride 16
        c5 = self.p5_refine(self.p5_proj(deep))       # stride 32

        return {"0": c2, "1": c3, "2": c4, "3": c5}

    def _forward_single_source(self, x):
        B, C, H, W = x.shape
        H_feat = H // self.patch_size
        W_feat = W // self.patch_size

        features_dict = self.backbone.forward_features(x)
        if "x_norm_patchtokens" in features_dict:
            patch_tokens = features_dict["x_norm_patchtokens"]  # [B, N, D]
        elif "x_patchtokens" in features_dict:
            patch_tokens = features_dict["x_patchtokens"]
        else:
            patch_tokens = list(features_dict.values())[0]

        B, N, D = patch_tokens.shape
        expected_N = H_feat * W_feat

        if N == expected_N + 1:
            patch_tokens = patch_tokens[:, 1:, :]
        elif N != expected_N:
            raise ValueError(
                f"Patch token count mismatch: expected {expected_N} (from image size {H}x{W}), got {N}. "
                f"Make sure image dimensions are divisible by patch size ({self.patch_size})."
            )

        feat_map = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, D, H_feat, W_feat)

        c4 = self.conv_c4(feat_map)
        c3 = self.deconv_c3(c4)
        c5 = self.pool_c5(c4)
        c6 = self.pool_c6(c5)

        return {"0": c3, "1": c4, "2": c5, "3": c6}

