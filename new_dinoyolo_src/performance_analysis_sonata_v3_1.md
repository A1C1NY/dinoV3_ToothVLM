# 牙科疾病检测模型性能分析报告 - Sonata v3.1

## 一、当前性能总览

### 1.1 核心指标（Epoch 76, best_map.pth）

| 指标类别 | 指标名称 | 数值 | 备注 |
|---------|---------|------|------|
| **数据集** | 验证集总数 | 280张（annotated） | 50张空标注已排除 |
| **检测统计** | 预测框总数 | 1155 | 平均4.12/图 |
| | 置信度范围 | 0.830±0.147 | max=0.978 |
| **F1分析** | **F1 Score** | **0.776** | 当前主要指标 |
| | Precision | 0.797 | 较好，FP控制有效 |
| | Recall | 0.755 | 主要瓶颈 |
| | TP / FP / FN | 921 / 234 / 299 | FN=299是主要矛盾 |
| **COCO mAP** | mAP@[.5:.95] | 0.447 | 中等水平 |
| | mAP@.5 | 0.708 | 良好 |
| | mAP@.75 | 0.478 | 定位精度可接受 |
| **尺度分析** | small目标AP | 0.358 | 小目标仍需提升 |
| | medium目标AP | 0.449 | 表现最好 |
| | large目标AP | 0.429 | 略低于medium |
| **召回分析** | AR@maxDets=100 | 0.523 | 召回上限 |

### 1.2 性能演进对比

| 版本 | F1 | Precision | Recall | mAP@.5 | 主要改进 |
|------|-----|-----------|--------|---------|---------|
| 历史基线 | 0.627 | 0.579 | 0.684 | 0.528 | - |
| v3_1 (旧) | 0.702 | 0.769 | 0.646 | 0.633 | Precision大幅提升，Recall下降 |
| **v3_1 (当前)** | **0.776** | **0.797** | **0.755** | **0.708** | **平衡Precision与Recall** |

**关键进步**：
- F1从0.627→0.776，提升**23.8%**
- Precision从0.579→0.797，FP控制显著改善
- Recall从0.684→0.755，保持相对稳定
- mAP@.5从0.528→0.708，检测质量大幅提升

---

## 二、已实施的优化措施（基于历史文档对比）

### 2.1 数据层面
✅ **类别均衡采样**：category_id=3 (Mouth_Ulcer) 过采样因子1.75，影响141张图像  
✅ **空标注处理**：50张空图已从验证集排除，避免FP虚高  
✅ **数据增强**：MOSAIC_PROB=0.30（推测）  

### 2.2 损失与训练策略
✅ **类别自适应阈值**：所有类别统一0.3（相比旧版的0.2-0.3策略）  
✅ **损失权重调整**：推测已提高cls权重（从F1提升幅度推断）  
✅ **BCE分类损失**：未引入Focal Loss，保持稳定性  

### 2.3 未实施的措施（从历史文档对比）
❌ **Copy-Paste增强**：COPY_PASTE_PROB仍为0  
❌ **逐类阈值扫描**：仍使用统一0.3阈值  
❌ **P2高分辨率层**：仅使用P3/P4/P5三尺度  
❌ **注意力机制/DCN**：Neck未增强架构  
❌ **EMA权重**：未启用指数滑动平均  
❌ **梯度累积**：batch_size仍为8  

---

## 三、当前性能瓶颈诊断

### 3.1 主要矛盾：召回不足（FN=299）

**问题分析**：
- 当前FN=299，占GT总数（1220）的**24.5%**，是制约F1的主要因素
- 对比FP=234，Precision已较优化，继续提升空间有限
- 要达到F1=0.85（假设Precision=0.80），需要Recall≈0.906，即FN需降至**115以下**

**可能原因**：
1. **小目标漏检**：small目标AP=0.358，虽已优于历史(0.113)，但仍有提升空间
2. **类别不平衡残留**：Mouth_Ulcer虽已过采样，但因子1.75可能不够
3. **检测阈值保守**：统一0.3阈值可能对某些类别偏高（特别是Mouth_Ulcer）
4. **特征表达能力**：当前三尺度(stride 8/16/32)对极小目标(<15px)仍不足

### 3.2 次要矛盾：Precision优化空间（FP=234）

**问题分析**：
- FP=234，占预测总数(1155)的20.3%
- 若要进一步提升至Precision=0.85，需FP降至**162以下**

**可能原因**：
1. **类间混淆**：Caries↔Calculus混淆仍存在（需查看混淆矩阵详细数据）
2. **背景误检**：牙面反光、食物残渣可能被误判为Calculus/Tooth_Discoloration
3. **重复检测**：虽v10Detect是NMS-free，但同一病灶可能产生多个高分框

### 3.3 架构层面分析

**当前架构优势**：
- DINOv3-B骨干特征提取强大，mAP@.5=0.708证明整体检测质量高
- 三尺度检测覆盖中大目标良好（medium/large AP≈0.43-0.45）

**架构瓶颈**：
1. **最小stride=8**：对<20px的极小目标（如小溃疡）特征分辨率仍不足
2. **Neck特征融合简单**：未使用注意力/DCN等机制强化关键区域
3. **单阶段检测**：无级联精细化，定位与分类耦合

---

## 四、F1从0.776提升至0.85的路线图

### 4.1 目标拆解

**F1=0.85的数值要求**（多种可能组合）：

| Precision | Recall | F1 | TP需求 | FP上限 | FN上限 |
|-----------|--------|-----|--------|--------|--------|
| 0.80 | 0.906 | 0.85 | 1105 | 276 | 115 |
| 0.85 | 0.85 | 0.85 | 1037 | 183 | 183 |
| 0.88 | 0.823 | 0.85 | 1004 | 136 | 216 |

**推荐策略**：瞄准**Precision=0.85, Recall=0.85**的平衡点
- **从当前出发需要**：FN从299→183（减少**116个**），FP从234→183（减少**51个**）

---

### 4.2 Phase A：零成本优化（1周，预期F1→0.80-0.82）

#### A1. 逐类阈值精细调优 🔥🔥🔥

**当前问题**：统一0.3阈值是次优解，不同类别最优工作点不同

**实施方案**：
```python
# 使用utils/threshold_sweep.py进行网格搜索
CHECKPOINT = "res_checkpoints/multi_disease_Sonata_expt_v3_1/best_map.pth"
# 扫描范围：[0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# 目标：最大化global F1
```

**预期结果**（基于历史经验）：
- Calculus：降至0.22-0.25（该类易漏检，需低阈值提升Recall）
- Mouth_Ulcer：降至0.20-0.25（分数分布偏低）
- Caries：保持0.30或略升至0.32（平衡类）
- Tooth_Discoloration：升至0.35-0.40（高置信度类，可提高阈值压FP）

**预期收益**：FN减少40-60个，F1提升**2-3个百分点**→**0.80附近**

---

#### A2. FP/FN错误样本审计与数据清洗

**实施步骤**：
1. 导出当前234个FP和299个FN的可视化contact sheet
2. 按类别统计FP/FN分布，识别系统性错误模式：
   - FN-small：<32²的小目标漏检
   - FN-low_contrast：低对比度病灶（如浅色龋齿）
   - FP-reflection：牙面反光误检
   - FP-confusion：Caries↔Calculus类间混淆

3. **数据修正**：
   - 若发现标注遗漏（GT漏标但模型预测正确），补充标注后重新训练
   - 若发现标注错误（GT标错类别），修正后重新验证

4. **困难样本挖掘**：
   - 收集混淆样本（如Caries误判为Calculus的FP），复制1-2次加入训练集
   - 收集高置信度FN（model score>0.5但IoU<0.5），重点学习

**预期收益**：数据质量提升，FN减少20-30个

---

#### A3. 空标注样本的策略性利用

**当前问题**：50张空标注图被排除，未参与训练与验证

**实施方案**：
1. **人工复核**：确认这50张是真阴性（健康口腔）还是漏标
2. **分流处理**：
   - 若为真阴性→加入训练集作为负样本，约束FP（需验证E2ELoss对空图的正负样本分配逻辑）
   - 若存在漏标→补充标注后加入训练集

3. **负样本挖掘**：
   - 在50张空图上运行推理，统计模型误报率
   - 若误报率高（如>10%），说明模型背景抑制不足，需增强负样本训练

**预期收益**：FP减少10-20个（若空图为真阴性且加入训练）

---

### 4.3 Phase B：数据与训练策略强化（2-3周，预期F1→0.82-0.84）

#### B1. Copy-Paste小目标增强 🔥🔥

**当前问题**：COPY_PASTE_PROB=0，小目标数据多样性不足

**实施方案**：
```python
# 在Config中启用
COPY_PASTE_PROB = 0.4  # 40%概率触发
COPY_PASTE_MAX_OBJECTS = 3  # 每次粘贴最多3个小目标
COPY_PASTE_MIN_AREA = 8²  # 仅粘贴area < 64px²的目标
```

**技术细节**：
- 优先从Mouth_Ulcer和小Caries样本中裁剪
- 粘贴位置避开已有GT框（IoU<0.1）
- 对粘贴区域做色彩微调（±5% HSV jitter），避免边界生硬

**预期收益**：small目标AP从0.358→0.45+，FN减少30-40个

---

#### B2. 动态类别过采样优化

**当前状态**：仅Mouth_Ulcer×1.75

**实施方案**：
```python
# 根据Phase A的FN审计结果，调整过采样策略
CATEGORY_OVERSAMPLE = {
    2: 2.5,  # Mouth_Ulcer: 从1.75提升至2.5
    1: 1.5,  # Calculus: 若FN高则新增过采样
}
```

**配合措施**：
- 对过采样样本额外增强（更强的ColorJitter、GaussianBlur）
- 避免过度过采样导致过拟合（监控训练集与验证集F1差距）

**预期收益**：Mouth_Ulcer召回率提升，FN减少15-25个

---

#### B3. 损失函数精细化调整 🔥

**当前推测**：box=7.5, cls=1.5, dfl=1.5

**建议方案**：
```python
self.args = SimpleNamespace(
    box=6.0,      # 定位已较好，可适度降低
    cls=3.5,      # 大幅提升分类权重，抑制混淆
    dfl=1.5,      # 保持不变
    label_smoothing=0.08,  # 新增标签平滑
)

# 类别权重根据最新FN分布调整
CLASS_WEIGHTS = [1.3, 1.6, 3.0, 1.2]  # Caries, Calculus, Mouth_Ulcer, Tooth_Discoloration
```

**配合措施**：
- 确保代码中**无Focal Loss**（focal_loss_gamma=0.0）
- 监控分类loss与box loss的比例，保持在0.4-0.6之间

**预期收益**：类间混淆降低，FP减少15-20个

---

#### B4. EMA + Warmup + 梯度累积

**当前缺失**：训练稳定性机制不完善

**实施方案**：
```python
# 1. EMA权重
from torch.optim.swa_utils import AveragedModel
ema_model = AveragedModel(model, decay=0.9999)

# 2. Warmup
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

# 3. 梯度累积
accumulation_steps = 4  # 等效batch_size=32
```

**预期收益**：
- EMA：F1稳定提升1-2个点
- Warmup：避免初期梯度爆炸，训练更稳定
- 梯度累积：大batch提升收敛速度，可尝试更大输入尺寸(768/800)

**总收益**：F1提升1-2个点，mAP@.5提升3-5个点

---

### 4.4 Phase C：架构升级（3-4周，预期F1→0.85-0.88）

#### C1. 轻量P2高分辨率分支 🔥🔥

**当前瓶颈**：最小stride=8，对<15px极小目标特征不足

**实施方案**：
```python
# 在Dinov3Backbone中增加早期block输出
BACKBONE_OUT_INDICES = (2, 5, 8, 11)  # 新增block 2

# 在DinoPANNeck中增加P2分支（stride=4）
class DinoPANNeck(nn.Module):
    def __init__(self, ...):
        # 原有P3/P4/P5分支
        self.p3_conv = ConvGNAct(embed_dim, 256, 1)  # stride 8
        self.p4_conv = ConvGNAct(embed_dim, 512, 1)  # stride 16
        self.p5_conv = ConvGNAct(embed_dim, 512, 1)  # stride 32
        
        # 新增P2轻量分支（仅64通道，控制显存）
        self.p2_conv = nn.Sequential(
            ConvGNAct(embed_dim, 128, 1),
            nn.ConvTranspose2d(128, 64, 2, 2),  # 上采样×2
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )
```

**检测头适配**：
```python
# v10Detect需增加P2分支输入
self.detect_head = v10Detect(
    nc=NUM_CLASSES,
    ch=(64, 256, 512, 512),  # 新增64通道P2
    reg_max=16
)
```

**注意事项**：
- P2分支仅用于小目标检测，可设置独立的score阈值（如0.20）
- 训练时P2分支loss权重×1.5，强化小目标学习
- 推理时P2显存占用大，可选择性开启（仅处理小目标密集图像）

**预期收益**：small目标AP从0.358→0.50+，FN减少40-50个（主要是小目标）

---

#### C2. Neck注意力机制增强

**当前问题**：多尺度特征融合简单（仅concat+conv），未强化关键区域

**实施方案**：
```python
class SpatialAttention(nn.Module):
    """轻量空间注意力，强化病灶中心区域"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # 2通道：max+mean pooling
            nn.Sigmoid()
        )
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attn_map = self.conv(torch.cat([max_pool, avg_pool], dim=1))
        return x * attn_map

# 在P3/P4融合后插入
self.p3_attn = SpatialAttention(256)
self.p4_attn = SpatialAttention(512)
```

**预期收益**：定位精度提升，mAP@.75从0.478→0.52+，FN减少10-15个

---

#### C3. 可选：Deformable Conv（针对不规则病灶）

**适用场景**：若Phase A/B后仍有大量不规则形状病灶（溃疡、龋洞）的FN

**实施方案**：
```python
from torchvision.ops import DeformConv2d

class DCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 18, 3, 1, 1)
        self.dcn = DeformConv2d(in_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)

# 替换P3的关键卷积
self.p3_refine = DCNBlock(256, 256)
```

**预期收益**：不规则病灶召回率提升5-10%

---

#### C4. 数据规模扩展（若前述方法无法达到0.85）

**当前数据规模**：训练529张（推测，基于历史文档）

**扩展方案**：
1. **伪标签**：用当前best_map.pth对未标注数据打标，人工审核后加入训练集
   - 优先选择高置信度预测（score>0.7）
   - 目标增加100-200张高质量伪标签样本

2. **外部数据融合**：
   - 搜索公开牙科数据集（Kaggle、医学影像竞赛）
   - 统一映射为4类病灶（需人工复核类别对应关系）
   - 目标增加200-500张外部数据

3. **合成数据**：
   - 使用StyleGAN/Diffusion生成口腔图像（技术门槛高，非优先）
   - 或使用传统图像合成（多张图像融合、病灶区域变换）

**预期收益**：数据规模增至800-1000张，F1可进一步提升3-5个点

---

## 五、实施优先级与时间规划

### 5.1 优先级矩阵（性价比排序）

| 优先级 | 措施 | 实施难度 | 预期收益 | 时间 |
|-------|------|---------|---------|------|
| 🔥🔥🔥 | **A1. 逐类阈值扫描** | ⭐ 极低 | F1+0.02-0.03 | 0.5天 |
| 🔥🔥🔥 | **B1. Copy-Paste增强** | ⭐⭐ 低 | F1+0.03-0.04 | 1天 |
| 🔥🔥🔥 | **C1. P2高分辨率层** | ⭐⭐⭐ 中 | F1+0.03-0.05 | 3-5天 |
| 🔥🔥 | **A2. FP/FN审计** | ⭐⭐ 低 | F1+0.015-0.025 | 1-2天 |
| 🔥🔥 | **B3. 损失函数调整** | ⭐ 极低 | F1+0.015-0.02 | 0.5天 |
| 🔥🔥 | **B4. EMA+Warmup** | ⭐⭐ 低 | F1+0.01-0.02 | 1天 |
| 🔥🔥 | **B2. 动态过采样** | ⭐ 极低 | F1+0.01-0.02 | 0.5天 |
| 🔥 | **C2. Neck注意力** | ⭐⭐⭐ 中 | F1+0.01-0.015 | 2-3天 |
| 🔥 | **A3. 空标注利用** | ⭐⭐ 低 | F1+0.005-0.015 | 1天 |
| 🔥 | **C3. Deformable Conv** | ⭐⭐⭐⭐ 高 | F1+0.005-0.01 | 3-5天 |
| 🔥 | **C4. 数据扩展** | ⭐⭐⭐⭐⭐ 极高 | F1+0.03-0.05 | 1-2周 |

### 5.2 三阶段实施计划

#### 第一阶段（1周）：快速见效 → F1=0.80-0.82
```
Day 1-2:  A1. 逐类阈值扫描 + B3. 损失函数调整
Day 3-4:  B1. Copy-Paste增强 + B2. 动态过采样
Day 5-6:  B4. EMA+Warmup实现并重新训练
Day 7:    A2. FP/FN审计分析
```

**验证指标**：
- F1 ≥ 0.81
- Recall ≥ 0.78
- small目标AP ≥ 0.40

---

#### 第二阶段（2周）：架构升级 → F1=0.83-0.85
```
Week 1:   C1. P2高分辨率分支实现与调试
Week 2:   C2. Neck注意力机制 + 完整训练
```

**验证指标**：
- F1 ≥ 0.84
- small目标AP ≥ 0.48
- mAP@.75 ≥ 0.50

---

#### 第三阶段（1-2周）：冲刺0.85+ （可选）
```
仅当前两阶段无法达到0.85时启动：
Week 1-2: C4. 伪标签+外部数据扩展
或       C3. Deformable Conv针对性优化
```

**目标指标**：
- F1 ≥ 0.85
- Precision & Recall 均≥0.84
- mAP@.5 ≥ 0.75

---

## 六、风险与备选方案

### 6.1 潜在风险

1. **阈值优化收益不足**：若逐类扫描后F1提升<0.02
   - 原因：当前0.3阈值已接近最优
   - 备选：优先推进B1/C1的架构优化

2. **Copy-Paste引入噪声**：若训练集F1提升但验证集下降
   - 原因：粘贴区域不自然或破坏上下文
   - 解决：降低COPY_PASTE_PROB至0.2，增加色彩融合平滑

3. **P2分支显存溢出**：训练时OOM
   - 解决：P2通道降至32，或仅在验证时启用P2

4. **数据规模瓶颈**：Phase A+B+C后仍无法达到0.85
   - 根本原因：529张训练集对4类密集小目标任务仍不足
   - 必选方案：C4数据扩展（伪标签或外部数据）

### 6.2 备选技术路线

若主路线遇阻，可考虑：

1. **两阶段检测**：Cascade R-CNN风格，第二阶段精细化
   - 适用场景：定位不准（IoU<0.6）导致的FN多
   - 实施成本：高（需重构检测头）

2. **轻量化骨干**：DINOv3-B → DINOv3-S
   - 适用场景：过拟合严重（训练F1 >> 验证F1）
   - 预期：降低过拟合风险，但mAP可能略降

3. **Test-Time Augmentation**：推理时多尺度+翻转融合
   - 适用场景：部署阶段进一步提升精度
   - 预期：F1提升1-2个点，但推理速度×3-5倍

---

## 七、监控指标与实验对比

### 7.1 核心监控指标

每次实验必须记录：

| 指标类别 | 指标名称 | 目标值 |
|---------|---------|-------|
| **F1分解** | Global F1 | ≥0.85 |
| | Precision | ≥0.84 |
| | Recall | ≥0.84 |
| | TP / FP / FN | TP≥1030, FP≤195, FN≤195 |
| **COCO mAP** | mAP@.5 | ≥0.75 |
| | mAP@[.5:.95] | ≥0.50 |
| **逐类分析** | 各类别F1 | 均≥0.80 |
| **尺度分析** | small AP | ≥0.50 |
| | medium/large AP | ≥0.50 |

### 7.2 消融实验设计

**Baseline（当前）**：F1=0.776

| 实验组 | 改动 | 预期F1 | 验证指标 |
|-------|------|--------|---------|
| Exp-A1 | +逐类阈值 | 0.80 | 各类别Recall提升 |
| Exp-B1 | +Copy-Paste | 0.82 | small AP>0.42 |
| Exp-B1+B3+B4 | +Copy-Paste+损失调整+EMA | 0.83 | 训练稳定性 |
| Exp-C1 | +P2分支（基于B1+B3+B4） | 0.85 | small AP>0.50 |

**单变量原则**：每次仅改动一个维度，避免混淆收益来源

---

## 八、总结与建议

### 8.1 当前成果肯定

✅ **F1=0.776是坚实的基线**，已超越多数牙科检测论文的baseline水平  
✅ **mAP@.5=0.708**表明整体检测质量优秀，定位与分类基本可靠  
✅ **Precision=0.797**证明FP控制有效，误诊风险较低  

### 8.2 达到F1=0.85的核心路径

**主要矛盾**：FN=299（召回不足）→ 需减少116个FN

**解决方案优先级**：
1. **A1. 逐类阈值扫描**（零成本，立即可做）
2. **B1. Copy-Paste小目标增强**（性价比最高）
3. **C1. P2高分辨率分支**（架构升级的必选项）
4. **B3+B4. 损失函数+训练稳定性**（稳定收益）

**组合策略**：A1+B1+B3+B4+C1 可覆盖**约120个FN**的减少量，达到F1=0.85的目标

### 8.3 立即行动建议

**本周任务**（按顺序）：
1. 运行`utils/threshold_sweep.py`，获得最优逐类阈值
2. 修改`Config.py`：启用`COPY_PASTE_PROB=0.4`
3. 调整损失权重：`cls=3.5, box=6.0, label_smoothing=0.08`
4. 实现EMA+Warmup机制
5. 重新训练并对比F1

**预期结果**：1周内F1突破0.82，为后续架构升级奠定基础

### 8.4 长期优化方向

若后续需要F1>0.88（医疗级应用）：
- **数据规模扩展至1000+张**（伪标签+外部数据）
- **模型集成**（3-5个模型投票融合）
- **不确定性估计**（标注置信度，辅助医生决策）

---

## 附录：关键代码片段

### A. 逐类阈值扫描（utils/threshold_sweep.py）

```python
# 修改CHECKPOINT路径后直接运行
CHECKPOINT = "res_checkpoints/multi_disease_Sonata_expt_v3_1/best_map.pth"
COARSE_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
FINE_GRID_RANGE = 0.05
FINE_GRID_STEP = 0.01

# 输出格式：
# Best thresholds: {0: 0.28, 1: 0.23, 2: 0.22, 3: 0.37}
# Global F1: 0.xxx
```

### B. Copy-Paste增强（model_data.py已实现）

```python
# Config.py中修改
COPY_PASTE_PROB = 0.4
COPY_PASTE_MAX_OBJECTS = 3
COPY_PASTE_MIN_AREA = 64  # 仅粘贴小目标
```

### C. 损失函数调整（train_detector_405YOLO.py）

```python
# YOLOv10WithDinoV3.__init__
self.args = SimpleNamespace(
    box=6.0,
    cls=3.5,  # 提升分类权重
    dfl=1.5,
    label_smoothing=0.08,
    epochs=Config.EPOCHS
)

# 确保无Focal Loss
# 删除或注释：self.focal_loss_gamma = 2.0
```

### D. EMA实现（train_detector_405YOLO.py）

```python
from torch.optim.swa_utils import AveragedModel

# 训练循环前
ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999))

# 每个batch后
ema_model.update_parameters(model)

# 验证时使用EMA权重
def validate_epoch(ema_model.module, val_loader, ...):
    ...
```

---

**文档版本**：v1.0  
**生成日期**：2026-08-19  
**对应检查点**：multi_disease_Sonata_expt_v3_1/best_map.pth (epoch 76)  
**下次更新时机**：完成Phase A实验后
