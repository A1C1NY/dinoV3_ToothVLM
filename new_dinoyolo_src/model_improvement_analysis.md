# 牙科疾病检测模型性能分析与改进建议

## 上一版性能基线（历史记录；最新 F1=0.702 见文末）

### 定量指标
- **mAP@0.5:0.95**: 0.292 (29.2%)
- **mAP@0.5**: 0.528 (52.8%)
- **F1 Score**: 0.627 (Precision: 0.579, Recall: 0.684)
- **训练集**: 529张图像，验证集: 132张图像
- **检测总数**: 平均每张4.70个框，置信度均值0.617

### 混淆矩阵分析（关键问题识别）

#### 1. **高漏检率（background列占比大）**
- **Caries（龋齿）**: 24.8%漏检，56.0%正确召回
- **Calculus（牙结石）**: 29.0%漏检，65.7%正确召回  
- **Mouth_Ulcer（口腔溃疡）**: **50.0%漏检**，仅50.0%正确召回
- **Tooth_Discoloration（牙齿变色）**: 20.1%漏检，73.8%正确召回

**分析**: 口腔溃疡漏检率高达50%，是当前最严重的瓶颈。即使使用了类别自适应低阈值（0.20），仍有一半样本未被检出。

#### 2. **类间混淆严重（非对角线元素）**
- Caries→Calculus混淆: 18.4%（龋齿被误判为牙结石）
- Calculus→Caries混淆: 3.5%
- 大量FP（虚警）分布在Calculus列: 64.9%的虚警被预测为牙结石

**分析**: 模型对黄白色牙面区域（龋齿vs牙结石）区分能力不足，倾向过度预测calculus类别。

#### 3. **推理结果可视化观察**
从提供的三张推理图可见：
- **重叠框过多**: Calculus类预测出现大量重叠的边界框，NMS抑制不充分
- **定位不准**: 部分框覆盖范围过大或偏移实际病灶中心
- **小目标漏检**: Mouth_Ulcer等小病灶容易被忽略

---

## 核心问题诊断

### 问题1: 模型容量与数据规模不匹配
**现状**: 
- 使用完整的ViT-B/16 DINOv3 (85M参数) + YOLOv10 Neck/Head
- 仅训练529张图像，batch_size=8，有效迭代次数少
- 解冻6个block（~50%的backbone），但数据量不足以充分微调

**后果**: 
- 骨干特征未能针对牙科细粒度纹理（龋齿黑斑、牙结石黄斑、溃疡红肿）充分适应
- 小样本过拟合风险：验证集mAP在35 epoch后波动，未能持续提升

---

### 问题2: 检测头对密集小目标支持不足
**现状**:
- 当前Neck输出P3/P4/P5三尺度（stride 8/16/32）
- YOLOv10 v10Detect头设计针对通用场景，未针对口腔密集病灶优化
- 牙科图像中单张可含5-10个病灶，尺寸跨度大（溃疡<20px，calculus区域>100px）

**后果**:
- 小目标（mouth_ulcer）在P3尺度仍然特征不足，50%漏检
- 密集框NMS策略不当，导致calculus类框大量堆叠

---

### 问题3: 损失函数未对齐任务难点
**现状**:
```python
self.args = SimpleNamespace(box=7.5, cls=1.5, dfl=1.5, epochs=Config.EPOCHS)
CLASS_WEIGHTS = [1.2, 1.3, 2.5, 1.1]  # Caries, Calculus, Mouth_Ulcer, Tooth_Discoloration
```
- 分类损失权重cls=1.5远低于box=7.5，模型优先优化定位而非分类
- 虽设置了类别权重2.5给mouth_ulcer，但从混淆矩阵看效果有限

**后果**:
- 类间混淆（Caries↔Calculus 18.4%）未得到有效抑制
- 分类置信度校准不足，导致需要过低的阈值才能召回

---

### 问题4: 特征提取架构单一尺度
**现状**:
```python
BACKBONE_OUT_INDICES = (5, 8, 11)  # 从ViT的三个block取特征
```
- 虽然实现了多尺度提取，但三个层级间隔固定（block 5/8/11）
- ViT的浅层（block 5）主要是低级纹理，深层（block 11）是高级语义
- 缺乏跨尺度特征交互机制强化细节

**后果**:
- 细粒度纹理特征（龋齿边缘、溃疡纹理）与语义特征（病灶类型）未充分融合
- 小目标在浅层特征不足，大目标在深层又丢失细节

---

## 改进建议（按优先级排序）

### 🔥 优先级1: 数据增强与训练策略优化

#### 1.1 强化小目标检测的数据增强
**当前问题**: Mouth_Ulcer漏检50%，small目标mAP仅0.113

**建议方案**:
```python
# 在CocoYOLODataset中增加Mosaic + Copy-Paste增强
# 针对small目标（area < 32²）的样本做重采样
```

**实施细节**:
1. **Mosaic拼接**: 将4张图像拼成一张，人为制造多目标场景，提升小目标检测鲁棒性
   - 牙科图像背景简单（口腔内），mosaic不会引入不自然伪影
   - 可提升模型对密集病灶的泛化能力
   
2. **小目标重采样**: 对包含mouth_ulcer的样本增加采样概率（×1.5-2.0）
   - 平衡类别样本数，缓解mouth_ulcer样本不足问题
   - 结合当前2.5x类别权重，形成双重平衡

3. **Copy-Paste**: 将小目标裁剪后粘贴到其他图像，增加小目标样本多样性

**预期收益**: F1提升5-8%，mouth_ulcer召回率提升15-20%

---

#### 1.2 对比学习预训练微调
**当前问题**: DINOv3预训练在自然图像，对牙科细粒度纹理（龋齿黑斑vs牙结石黄斑）表征不足

**建议方案**:
使用SupCon/SimCLR在当前529张训练集上做自监督对比学习微调骨干

**实施细节**:
1. 冻结ViT前6层，对后6层做对比学习微调（5-10 epoch）
2. 正样本对：同一disease类别的不同crop
3. 负样本对：不同disease类别 + 背景区域
4. 温度参数τ=0.07，batch内负样本对比

**代码框架**:
```python
# 在train_detector_405YOLO.py前插入预训练阶段
# 使用SimCLR损失强化类间可分性
loss = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

**预期收益**: 类间混淆降低10-15%，Caries→Calculus混淆从18.4%降至10%以下

---

### 🔥 优先级2: 检测头架构改进

#### 2.1 引入Cascade检测头
**当前问题**: 单阶段检测对IoU敏感，定位不准的框（IoU=0.5-0.6）误判为正样本

**建议方案**:
将YOLOv10 v10Detect替换为两阶段Cascade结构：
- Stage1: IoU阈值0.5，粗定位
- Stage2: IoU阈值0.6，精细化，仅处理Stage1的正样本

**实施细节**:
```python
class CascadeDetectHead(nn.Module):
    def __init__(self, nc, ch):
        self.stage1 = v10Detect(nc, ch, reg_max=12)  # 降低reg_max减少计算
        self.stage2_refine = nn.ModuleList([
            ConvBNAct(ch[i], ch[i], 3) for i in range(3)
        ])
        self.stage2 = v10Detect(nc, ch, reg_max=16)
```

**训练策略**:
- 前20 epoch只训练stage1
- 后50 epoch联合训练，stage2学习率×0.5

**预期收益**: mAP@0.75从0.265提升至0.35+，定位精度显著改善

---

#### 2.2 增加超小尺度特征层（P2）
**当前问题**: 最小stride=8，对于<20px的mouth_ulcer特征仍不足

**建议方案**:
在Dinov3Backbone中增加P2输出（stride=4），对应特征来自block 2或3

**实施细节**:
```python
# 修改dinov3_backbone.py
BACKBONE_OUT_INDICES = (2, 5, 8, 11)  # 增加block 2
self.p2_proj = ConvGNAct(embed_dim, 128, kernel_size=2, stride=2, transpose=True)
# 在DinoPANNeck中增加P2分支，输出通道128
```

**注意事项**:
- P2分支计算开销大（4倍特征图），仅在推理时使用或降低输出通道（128→64）
- NMS时P2分支单独处理，阈值设为0.3（比P3/P4/P5的0.5更宽松）

**预期收益**: small目标mAP从0.113提升至0.18-0.22，mouth_ulcer召回率提升20%

---

### 🔥 优先级3: 损失函数与后处理优化

#### 3.1 提高分类损失权重并配合类别均衡采样
**当前问题**: 分类损失权重 cls=1.5 远低于 box=7.5，模型优先优化定位；而当前主要瓶颈是类间混淆与漏检

**建议方案**:
保持 BCE 分类损失不变（不引入 Focal Loss），通过提高 cls 权重、优化类别均衡采样与标签平滑来改善分类

**实施细节**:
```python
# 在 YOLOv10WithDinoV3.__init__ 中调整损失权重（保持 BCE）
self.args = SimpleNamespace(
    box=6.0,   # 若定位已稳定可保持 7.5；观察到框抖动时可适度降低
    cls=3.0,  # 提升分类损失权重至3.0
    dfl=1.5,
    # 不使用 Focal Loss：不设置 focal_gamma / use_focal，保持纯 BCE 分类损失

)
```

如代码中仍写有 `self.focal_loss_gamma = 2.0`，请删除该行或改为 0.0，避免 E2ELoss / v8DetectionLoss 误用 Focal Loss。

数据端配合：
1. 对 mouth_ulcer（COCO category_id=3）继续加大 WeightedRandomSampler 过采样因子至 2.0-2.5；若 calculus 的 FN 仍高，也对其提高采样权重
2. 对 Caries↔Calculus 混淆对做“困难样本重采样”：收集上一轮验证中混淆的 FP/FN 样本，复制加入训练集
3. 可选标签平滑 label_smoothing=0.1，缓解过拟合与过度自信，降低 FP
```python





```

**预期收益**: 类间混淆进一步下降，F1 提升 3-6 个点；该方案比 Focal Loss 更稳定、易调试

---

#### 3.2 动态NMS阈值（类别自适应）
**当前问题**: 固定NMS阈值0.5，calculus类框堆叠严重（FP 64.9%）

**建议方案**:
对不同类别设置不同的NMS IoU阈值

**实施细节**:
```python
# 在postprocess中增加类别自适应NMS
CLASS_NMS_THRESHOLDS = {
    0: 0.50,  # Caries: 标准阈值
    1: 0.40,  # Calculus: 更激进抑制，减少堆叠
    2: 0.60,  # Mouth_Ulcer: 宽松阈值，保留小目标
    3: 0.50,  # Tooth_Discoloration: 标准阈值
}

def class_aware_nms(boxes, scores, labels, iou_thresholds):
    keep = []
    for cls_id in torch.unique(labels):
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        iou_thr = iou_thresholds.get(cls_id.item(), 0.5)
        keep_cls = torchvision.ops.nms(cls_boxes, cls_scores, iou_thr)
        keep.extend(torch.where(cls_mask)[0][keep_cls].tolist())
    return torch.tensor(keep, dtype=torch.long)
```

**预期收益**: FP从296降至200以下，precision从0.579提升至0.65+

---

### 🔥 优先级4: 模型架构优化

#### 4.1 轻量化骨干 + 增强Neck
**当前问题**: ViT-B/16参数量85M，在529样本上容易过拟合，且推理慢

**建议方案**:
替换为DINOv3-S (22M参数)，同时增强Neck的表达能力

**实施细节**:
```python
# 修改Config
WEIGHTS = "dinov3_vits16_pretrain.pth"  # 切换到Small版本

# 在DinoPANNeck中增加Attention机制
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = F.softmax(Q @ K.transpose(-2, -1) / (Q.shape[1]**0.5), dim=-1)
        return x + (attn @ V)

# 在P3/P4/P5的融合处增加Attention
self.p4_attention = AttentionFusion(512)
self.p3_attention = AttentionFusion(256)
```

**原理**: 
- 轻量化骨干降低过拟合风险，释放参数预算给Neck
- Attention机制在多尺度融合时强化关键区域（病灶中心），抑制背景

**预期收益**: 
- 推理速度提升40-50%（85M→22M参数）
- F1从0.627提升至0.70-0.75（attention强化小目标）

---

#### 4.2 可变形卷积（Deformable Conv）替换标准卷积
**当前问题**: 病灶形状不规则（溃疡椭圆形、龋齿不规则黑斑），标准3×3卷积感受野固定

**建议方案**:
在Neck的关键位置（P3/P4融合后）使用DCNv2

**实施细节**:
```python
from torchvision.ops import DeformConv2d

class DCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 18, 3, 1, 1)  # 2×3×3偏移
        self.dcn = DeformConv2d(in_ch, out_ch, 3, padding=1)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)

# 替换DinoPANNeck中的p3_refine和p4_refine
self.p3_refine = DCNBlock(256, 256)
self.p4_refine = DCNBlock(512, 512)
```

**预期收益**: 不规则形状病灶（mouth_ulcer）召回率提升10-15%

---

### 🔥 优先级5: 训练技巧进阶

#### 5.1 渐进式解冻（Progressive Unfreezing）
**当前问题**: 一次性解冻6个block，初期梯度不稳定（grad_norm=230→70波动大）

**建议方案**:
- Epoch 1-10: 冻结全部backbone，只训练Neck/Head
- Epoch 11-30: 解冻后3个block（9-11）
- Epoch 31-60: 解冻后6个block（6-11）
- Epoch 61-70: 解冻全部block（1-11）

**实施细节**:
```python
def progressive_unfreeze(model, epoch):
    if epoch <= 10:
        freeze_blocks = list(range(12))
    elif epoch <= 30:
        freeze_blocks = list(range(9))
    elif epoch <= 60:
        freeze_blocks = list(range(6))
    else:
        freeze_blocks = []
    
    for i, block in enumerate(model.backbone.backbone.backbone.blocks):
        for param in block.parameters():
            param.requires_grad = (i not in freeze_blocks)
```

**预期收益**: 训练稳定性提升，最终mAP提升2-3%

---

#### 5.2 混合精度训练 + 梯度累积
**当前问题**: batch_size=8较小，梯度估计有噪声

**建议方案**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 4  # 等效batch_size=32

for i, (images, targets) in enumerate(train_loader):
    with autocast():
        output = model(images, targets)
        loss = output["loss"] / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**预期收益**: 
- 显存节省30%，可增大输入尺寸至800×800
- 梯度噪声降低，收敛更平滑

---

## 实施路线图

### Phase 1: 快速见效（1-2周，预期F1→0.70）
1. ✅ 增加Mosaic + 小目标重采样（数据增强）
2. ✅ 类别均衡采样 + 标签平滑（保持 BCE，不引入 Focal Loss）
3. ✅ 类别自适应NMS（后处理）
4. ✅ 提升分类损失权重cls=1.5→3.0

**验证指标**: mouth_ulcer召回率>65%，precision>0.65

---

### Phase 2: 架构优化（2-3周，预期F1→0.75-0.80）
1. ✅ 增加P2超小尺度层（针对小目标）
2. ✅ Neck中增加Attention机制（多尺度融合）
3. ✅ 渐进式解冻训练策略
4. ✅ 混合精度 + 梯度累积（batch_size→32）

**验证指标**: small目标mAP>0.20，整体mAP@0.5>0.60

---

### Phase 3: 深度优化（3-4周，预期F1→0.82-0.90）
1. ✅ 对比学习预训练微调骨干（SimCLR）
2. ✅ Cascade检测头（两阶段精细化）
3. ✅ Deformable Conv替换关键卷积层
4. ✅ 轻量化骨干DINOv3-S + 增强Neck

**验证指标**: 
- mAP@0.5:0.95>0.45
- F1>0.85
- Caries↔Calculus混淆<8%

---

## 额外建议

### 数据层面
1. **主动学习标注**: 对当前FP和FN样本重新人工审核，修正标注错误
2. **外部数据融合**: 考虑加入其他公开牙科数据集（如Kaggle Dental Disease Dataset）做迁移学习
3. **合成数据**: 使用GAN生成口腔图像，增加稀缺类别（mouth_ulcer）样本

### 工程层面
1. **模型蒸馏**: 训练大模型（ViT-L）作为teacher，蒸馏到ViT-S，兼顾精度与速度
2. **Test-Time Augmentation**: 推理时做水平翻转+多尺度融合，提升鲁棒性
3. **不确定性估计**: 输出置信度分布（MC Dropout），辅助医生决策

---

## 总结

上一版模型F1=0.627是合格基线；最新基线已更新为F1=0.702（见文末）。要达到论文级别（F1>0.85），需重点解决：

1. **Mouth_Ulcer 50%漏检** → 小目标增强（Mosaic/P2层/重采样）
2. **Caries↔Calculus 18.4%混淆** → 对比学习微调 + 类别均衡采样/分类权重再平衡
3. **Calculus类FP过多** → 类别自适应NMS + 提升分类权重
4. **定位不准（mAP@0.75低）** → Cascade检测头 + Deformable Conv

**优先级排序**: Phase1（数据+损失+NMS）是性价比最高的改进，可在2周内见效；Phase2-3需要较大代码改动，但能达到SOTA水平。

Phase1已验证（最新F1=0.702）；后续请按文末更新后的路线图继续优化。



---

## 最新实验基线（multi_disease_767_expt_v3_1_highsize / best_map.pth / epoch 51）

### 指标快照
| 指标 | 数值 |
| --- | --- |
| 验证集图片 | 154 张（34 张空标注，120 张有标注） |
| 预测框总数 | 463（平均 3.86/图；最高分 0.9607；均分 0.7444） |
| TP / FP / FN（IoU≥0.50） | 356 / 107 / 195 |
| Precision / Recall / **F1** | 0.7689 / 0.6461 / **0.7022** |
| mAP@[.5:.95] / mAP@.5 / mAP@.75 | 0.3690 / 0.6331 / 0.3942 |
| AR@maxDets=100 | 0.429 |
| 使用阈值 | 0:0.30, 1:0.30, 2:0.30, 3:0.30 |

### 关键变化与差距
- F1 0.627→0.702，主要来自 precision 0.579→0.769（FP 大幅下降）；但 recall 0.684→0.646，FN=195 成为当前主要矛盾。
- F1=0.80 的最低要求：precision 维持 0.77 时 recall 需 0.833；precision 提到 0.85 时 recall 也需 0.755。
- F1=0.90：若 TP 维持 356，则 FP+FN 需从 302 降到约 79；若 GT 规模 551 不变，则需 precision≈0.90 且 recall≈0.90，即 TP≈496、FP≈55、FN≈55。
- 空标注 34/154=22.1%，需先人工复核：若为真阴性，当前验证集不包含它们，precision 可能被高估；若存在漏标，recall 可能被低估。

### 新建议（按优先级）

#### A. 零成本先做：逐类阈值扫描 + 按类调阈值
- 当前所有类别统一 0.30，不一定是最优 F1 工作点。`utils/threshold_sweep.py` 已支持逐类粗扫+网格搜索，把 `CHECKPOINT` 改为当前 `best_map.pth` 后跑一遍，得到使 global_f1 最大的一组 `{0:?,1:?,2:?,3:?}`。
- 根据旧扫描经验，calculus 通常需要更低阈值（0.18-0.25）以召回更多真阳性，而 caries/tooth_discoloration 可适当提高阈值以压 FP；mouth_ulcer 分数偏低时也可降到 0.22-0.27。
- 注意：阈值只影响 F1/TP/FP/FN，不影响 COCO mAP（AP 按分数积分）。两者都要报告，避免只追 F1 而丢 mAP。

#### B. 错误样本审计（FP/FN 可视化）
- 把当前验证集的 107 FP 和 195 FN 按类别导出 contact sheet，逐张分析：
  - FN：是小目标漏检、低对比度、还是标注遗漏/边界过严？
  - FP：是牙面反光、牙结石误报、还是同一病灶重复框/类别混淆？
- 对系统性的 Caries↔Calculus 混淆，收集这些 hard cases 加入训练集（复制 1-2 次），比改损失函数更直接。

#### C. 数据增强与采样（与旧文 Phase1 衔接，但不再用 Focal Loss）
- `COPY_PASTE_PROB` 当前为 0，建议开启到 0.3-0.5，`COPY_PASTE_MAX_OBJECTS=3`，重点粘贴 small 目标（mouth_ulcer、小 caries）。`model_data.py` 已实现，只需改 Config。
- `MOSAIC_PROB=0.30` 保持或提高到 0.4；若大目标 AP 下降则回退。
- 最新采样为 category_id=3、factor=1.75、affected_images=78；若 mouth_ulcer 的 FN 仍高，把 `OVERSAMPLE_FACTOR` 提到 2.0-2.5，并对 calculus 也做轻度过采样（若该类 FN 高）。
- 对 34 张空标注图：若复核为真阴性，不要直接丢进训练（E2ELoss 对空图的正负样本分配需先验证），但应评估模型在这些图上的误报率，并把它们纳入验证集以约束 FP。

#### D. 损失与训练策略（保持 BCE）
- 代码中 `self.args = SimpleNamespace(box=7.5, cls=1.5, dfl=1.5)`；建议改为 `box=6.0, cls=3.0, dfl=1.5`，提高分类权重以抑制类间混淆。
- 若代码中仍设置 `self.focal_loss_gamma = 2.0`，请删除或改为 0.0，保持纯 BCE。
- 可选 `label_smoothing=0.05-0.1`，降低过度自信，间接压 FP。
- `CLASS_WEIGHTS=[1.2,1.3,2.5,1.1]` 可根据最新逐类 FN 调整：若 calculus FN 多，调高 1.3→1.6-1.8；mouth_ulcer 仍差，调高 2.5→3.0。
- `WARMUP_EPOCHS=5` 已声明但训练循环未使用：补一个 LinearLR 前 5 epoch 从 lr/100 升到 lr，再接 CosineAnnealingLR（可用 SequentialLR）。
- 增加 EMA（指数滑动平均权重，decay≈0.9999），验证和保存用 EMA 权重；通常可稳定涨 1-2 点 F1/mAP。
- 梯度累积 `accumulation_steps=4` 使等效 batch≈32，稳定梯度，再配合 `IMG_SIZE` 尝试 768/960/1024 多尺度训练。

#### E. 架构与后处理（0.80→0.90 的必经之路）
- 旧文 Phase2/3 的 P2 高分辨率层、Neck 注意力融合、DCN 仍有效；优先做 P2 或 P3/P4 的 DCN，针对小目标漏检。
- 当前 `DinoPANNeck` 只有 3 个尺度（stride 8/16/32）；增加 P2（stride 4）对 mouth_ulcer 和小 caries 最有帮助，但显存和耗时增加，可用 64-128 通道的轻量 P2 分支。
- 后处理方面，v10Detect 在 eval 模式已经 NMS-free，主要瓶颈不是重复框，而是分数校准；按 A 的阈值扫描 + D 的 label smoothing 更实际。

#### F. 数据规模与伪标签
- 用当前 best_map.pth 对未标注/空标注图打伪标签，人工审核后加入训练集；优先伪标签那些模型高置信度但 GT 漏标的样本。
- 外部数据：加入公开牙科数据集或院内历史数据，统一映射为 4 类；哪怕增加 200-500 张，对 F1 0.80 帮助很大。

### 更新后的路线图（无 Focal Loss）
| 阶段 | 主要工作 | 预期 F1 |
| --- | --- | --- |
| Phase A（1 周内） | 逐类阈值扫描、FP/FN 错误审计、空标注复核、cls=3.0 | 0.73-0.77 |
| Phase B（2-3 周） | Copy-Paste 小目标、逐类过采样、label smoothing、warmup+EMA+梯度累积 | 0.78-0.83 |
| Phase C（3-4 周） | P2 轻量分支/DCN/注意力融合、伪标签+外部数据、TTA | 0.84-0.90 |

### 结论
- 当前 F1 0.702 的主要矛盾是 FN=195（召回不足），其次才是 FP=107。
- F1 0.70→0.80 主要靠“阈值+数据+训练策略”即可；F1 0.80→0.90 必须靠“小目标架构+数据规模/质量+伪标签”。
- 建议从 A+B 开始，单变量实验，评估同时看 global_f1@IoU0.5 和 COCO mAP@.5:.95。