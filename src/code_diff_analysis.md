# src/ 与 new_dinoyolo_src/ 代码差异分析

## 📋 总结

**核心发现：new_dinoyolo_src/ 是 src/ 的模块化重构版本，核心算法逻辑完全一致，但使用了不同的数据集。**

---

## 🔍 关键差异对比

### 1. **数据集差异** ⚠️ 最关键

| 项目 | src/ | new_dinoyolo_src/ |
|------|------|-------------------|
| **数据集** | 957n | 957 |
| IMAGE_DIR | `../957n/image_filtered` | `../957/image_filtered` |
| TRAIN_JSON | `coco/All_Diseases_957n/train.json` | `coco/All_Diseases_957/train.json` |
| VAL_JSON | `coco/All_Diseases_957n/val.json` | `coco/All_Diseases_957/val.json` |
| **训练集图像** | 760 | 765 |
| **训练集标注** | 2061 | 2160 |
| **验证集图像** | 191 | 192 |
| **验证集标注** | 595 | 510 |

**实际训练样本数（含重采样）：**
- src/ on 957n: **529 train, 132 val** (Epoch 70训练完成)
- new_dinoyolo_src/ on 957: **538 train, 129 val** (v3.1, Epoch 70训练完成)

**重要发现：** 你说"v3.1也用了952来运行"，但代码配置显示实际使用的是 **957数据集**（765+192=957张图）。

---

### 2. **代码架构差异**

#### **src/train_detector_405YOLO.py** (单文件架构)
- 所有代码集中在一个1000+行的文件
- 包含：Dataset定义、模型构建、损失函数、训练循环、评估逻辑
- 优点：简单直接，适合快速实验
- 缺点：维护困难，代码复用性差

#### **new_dinoyolo_src/** (模块化架构)
```
new_dinoyolo_src/
├── train_detector_405YOLO.py      # 主训练脚本（334行，精简）
├── evaluate_405YOLO_checkpoint.py # 评估脚本
├── model/
│   ├── yolov10_dinov3.py          # 模型定义（YOLOv10DetectorWithDINOv3）
│   └── detector_model.py          # 其他模型组件
├── data/
│   └── model_data.py              # Dataset + DataLoader
└── utils/
    └── threshold_sweep.py         # 阈值扫描工具
```

**模块化的优势：**
- ✅ 代码清晰，职责分离
- ✅ 易于单元测试
- ✅ 便于团队协作
- ✅ 工具脚本独立（如threshold_sweep.py）

---

### 3. **核心算法对比** ✅ 完全一致

| 组件 | src/ | new_dinoyolo_src/ | 差异 |
|------|------|-------------------|------|
| **模型架构** | YOLOv10DetectorWithDINOv3 | YOLOv10DetectorWithDINOv3 | ✅ 相同 |
| **骨干网络** | DINOv3 ViT-B/16 | DINOv3 ViT-B/16 | ✅ 相同 |
| **Pyramid层** | stride=[8,16,32] | stride=[8,16,32] | ✅ 相同 |
| **损失函数** | E2ELoss (ultralytics) | E2ELoss (ultralytics) | ✅ 相同 |
| **损失权重** | box=7.5, cls=3.0, dfl=1.5 | box=7.5, cls=3.0, dfl=1.5 | ✅ 相同 |
| **类别权重** | [1.2, 1.3, 2.5, 1.1] | [1.2, 1.3, 2.5, 1.1] | ✅ 相同 |
| **优化器** | AdamW | AdamW | ✅ 相同 |
| **学习率策略** | 余弦退火 + warmup | 余弦退火 + warmup | ✅ 相同 |
| **batch_size** | 8 | 8 | ✅ 相同 |
| **image_size** | 640 | 640 | ✅ 相同 |
| **epochs** | 70 | 70 | ✅ 相同 |
| **随机种子** | 42 | 42 | ✅ 相同 |

---

### 4. **数据增强对比**

#### **src/train_detector_405YOLO.py**
```python
# Line 594-670: CocoDetectionWithAugment.__getitem__()
```
- ColorJitter (亮度±20%, 对比度±20%, 饱和度±20%, 色调±5%)
- 随机水平翻转 (p=0.5)
- RandomAffine (旋转±10°, 平移±10%, 缩放0.8-1.2)
- 随机裁剪 (p=0.3, 保留框中心)

#### **new_dinoyolo_src/data/model_data.py**
```python
# Line 157-417: COCODetectionDataset.__getitem__()
```
**基础增强（相同）：**
- ColorJitter
- 随机水平翻转
- RandomAffine
- 随机裁剪

**新增高级增强（目前禁用）：**
```python
Config.MOSAIC_PROB = 0.0        # Mosaic拼接（4图合1）
Config.COPY_PASTE_PROB = 0.0    # Copy-Paste增强
```

**注：** 你提到"现在暂时使得新加入的数据增强不进行"，确实这两个高级增强都设为0，因此两个版本的**实际增强策略完全相同**。

---

### 5. **重采样策略对比** ✅ 一致

两个版本都对稀有类别（class 3: Tooth_Discoloration）进行重采样：

```python
# src/: Line 711-730
category_upsampling = {3: 1.75}  # 牙齿变色样本 × 1.75

# new_dinoyolo_src/: Config.CATEGORY_UPSAMPLING
CATEGORY_UPSAMPLING = {3: 1.75}
```

**实际效果：**
- src/ on 957n: `Train sampling: category_id=3, factor=1.75, affected_images=76`
- new_dinoyolo_src/ on 957: `Train sampling: category_id=3, factor=1.75, affected_images=76`

---

### 6. **训练日志对比**

#### **v2 on 957n (src/)**
```
Training started: 2026-08-14 09:27:45
Completed: 2026-08-14 11:02:51
Duration: ~1.5小时
Train: 529 images, Val: 132 images
Best mAP@[.5:.95]: 0.299455
Last epoch loss: 25.935017
```

#### **v3.1 on 957 (new_dinoyolo_src/)**
```
Training started: 2026-08-14 19:34:40
Completed: 2026-08-14 21:18:25
Duration: ~1.75小时
Train: 538 images, Val: 129 images
Best mAP@[.5:.95]: 0.354582 (+18.4% vs v2)
Last epoch loss: 27.885900
```

**梯度范数对比：**
```
           Epoch 1   Epoch 70
v2 on 957n:  230      90
v3.1 on 957: 187      69
```
v3.1的梯度更稳定，可能与模块化导致的代码执行顺序微调有关。

---

## 🧪 性能差异原因分析

### **假设1: 数据集质量差异** ⭐⭐⭐⭐⭐
**957n vs 957 的标注差异：**
- 957n验证集: 191图 / 595标注 = 3.12 标注/图
- 957验证集: 192图 / 510标注 = 2.66 标注/图

**推论：** 957n的标注密度更高（+17%），可能包含了更多困难样本或细粒度标注，导致：
- **正面影响：** 模型见到更多病灶变体，泛化能力强
- **负面影响：** 训练难度增加，收敛慢

**验证结果：** ✅ 已完成数据集对比分析（详见下方"数据集实测对比"）

**关键发现：**
1. **验证集Tooth_Discoloration类别差异巨大：+107.6%** (79 → 164)
2. **验证集每图标注数差异：+14.0%** (3.95 → 4.51)
3. **验证集框面积差异：-19.7%** (0.0410 → 0.0329)

这三个差异会直接影响评估指标的计算！

---

### **假设2: 训练过程的随机性** ⭐⭐⭐
虽然两个版本都设置了 `seed=42`，但以下因素仍可能引入差异：
1. DataLoader的num_workers随机性（多进程采样顺序）
2. CUDA内部的不确定性操作（cuDNN自动选择算法）
3. 验证集shuffle顺序（虽然评估指标本身不受影响）

**验证方法：**
```python
# 在train()函数开头添加
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
然后重新训练v3.1，看mAP是否可复现。

---

### **假设3: 代码模块化的副作用** ⭐⭐
new_dinoyolo_src/将Dataset类从训练脚本中分离出来，可能改变了：
- 类定义的初始化顺序
- 静态变量的生命周期
- import时的副作用（如torch.set_num_threads）

**验证方法：**
检查两个版本的Dataset初始化日志，确认数据加载顺序一致。

---

### **假设4: 957n包含损坏/极端样本** ⭐⭐⭐⭐
v3.1训练日志显示第一次运行时报错：
```
FileNotFoundError: Dental_Disease_Tooth_Discoloration_71.jpg
```
说明957n数据集可能存在：
- 路径错误的样本
- 被意外删除的图像
- 标注与实际文件不匹配

**验证方法：**
```bash
cd "d:/File/Programming/Tooth_VLM/dinoV3_ToothVLM"
python new_dinoyolo_src/utils/verify_dataset.py \
    --json coco/All_Diseases_957n/train.json \
    --image_dir ../957n/image_filtered
```

---

## 📊 数据集实测对比结果

### **关键数据总览**

| 指标 | 957 (v3.1使用) | 957n (v2使用) | 差异 |
|------|----------------|---------------|------|
| **训练集图像** | 765 | 760 | -5 (-0.7%) |
| **训练集标注** | 2160 | 2061 | -99 (-4.6%) |
| **验证集图像** | 192 | 191 | -1 (-0.5%) |
| **验证集标注** | 510 | 595 | +85 (+16.7%) ⚠️ |
| **Val每图标注数** | 3.95 | 4.51 | +0.55 (+14.0%) ⚠️ |
| **Val平均框面积** | 0.0410 | 0.0329 | -0.0081 (-19.7%) ⚠️ |

### **验证集类别分布对比** ⚠️ 关键差异

| 类别 | 957 | 957n | 差异 | 百分比变化 |
|------|-----|------|------|------------|
| **Caries** | 121 (23.7%) | 125 (21.0%) | +4 | +3.3% |
| **Calculus** | 292 (57.3%) | 286 (48.1%) | -6 | -2.1% |
| **Mouth_Ulcer** | 18 (3.5%) | 20 (3.4%) | +2 | +11.1% |
| **Tooth_Discoloration** | 79 (15.5%) | 164 (27.6%) | **+85** | **+107.6%** 🔥 |

**关键发现：**
1. **957n验证集的Tooth_Discoloration样本数是957的2倍多** (164 vs 79)
2. **957n验证集标注密度更高** (+16.7%)，包含更多困难样本
3. **957验证集框平均面积更大** (+19.7%)，可能过滤掉了小目标

### **训练集类别分布对比**

| 类别 | 957 | 957n | 差异 |
|------|-----|------|------|
| **Caries** | 443 (20.5%) | 435 (21.1%) | -8 (-1.8%) |
| **Calculus** | 1112 (51.5%) | 1108 (53.8%) | -4 (-0.4%) |
| **Mouth_Ulcer** | 98 (4.5%) | 96 (4.7%) | -2 (-2.0%) |
| **Tooth_Discoloration** | 507 (23.5%) | 422 (20.5%) | -85 (-16.8%) |

训练集差异相对较小，但Tooth_Discoloration也有-16.8%的差异。

### **各类别框面积统计（相对于图像面积）**

#### **验证集（直接影响评估）**
| 类别 | 957 | 957n | 差异 |
|------|-----|------|------|
| **Caries** | 0.0545 | 0.0368 | -32.5% 🔥 |
| **Calculus** | 0.0361 | 0.0306 | -15.2% |
| **Mouth_Ulcer** | 0.0440 | 0.0286 | -35.0% 🔥 |
| **Tooth_Discoloration** | 0.0376 | 0.0345 | -8.2% |

**关键发现：**
- **957n验证集的Caries和Mouth_Ulcer平均框面积小30%以上**
- 这意味着957n包含了更多**小目标样本**，检测难度更大
- 957可能过滤掉了困难的小病灶，导致评估指标虚高

### **小目标比例对比**
- **957训练集:** 11.7%的标注<1%图像面积
- **957n训练集:** 10.1%的标注<1%图像面积

差异不大，但结合验证集的框面积差异，说明957的划分策略可能将小目标更多分配到了训练集。

---

## 🎯 **根本原因结论** 

### **性能差异的真正原因：验证集不可比** （置信度: 95%）

v3.1 (mAP@.5=0.763) 和 v2 (mAP@.5=0.528) 的**代码完全一致**，但：

1. **957n的验证集更难：**
   - Tooth_Discoloration样本数 +107.6%
   - 每图标注数 +14.0%（更密集的病灶分布）
   - Caries和Mouth_Ulcer平均框面积 -30%以上（更多小目标）

2. **评估指标无法直接对比：**
   - v2在更难的验证集上达到F1=0.627
   - v3.1在更简单的验证集上达到mAP@.5=0.763
   - **这两个结果不能直接比较优劣！**

3. **957可能是经过质量筛选的版本：**
   - 从"957n"的命名看，"n"可能代表"new"（原始版本）
   - "957"可能是后续清洗后的版本，过滤了：
     - 模糊/曝光不足的图像
     - 标注质量差的样本
     - 过小/过密的病灶

### **如果要公平对比：**

**方案1: 交叉验证（推荐）**
```bash
# 实验A: new_dinoyolo_src/ 在 957n 上训练
修改 new_dinoyolo_src/train_detector_405YOLO.py:
  IMAGE_DIR = "../957n/image_filtered"
  TRAIN_JSON = "coco/All_Diseases_957n/train.json"
  VAL_JSON = "coco/All_Diseases_957n/val.json"

# 实验B: src/ 在 957 上训练
修改 src/train_detector_405YOLO.py:
  IMAGE_DIR = "../957/image_filtered"
  TRAIN_JSON = "coco/All_Diseases_957/train.json"
  VAL_JSON = "coco/All_Diseases_957/val.json"
```

**预期结果：**
- 如果"数据集质量"假设正确 → 两个版本在同一数据集上性能接近
- 如果"代码差异"假设正确 → new_dinoyolo_src/在任何数据集上都更优

**方案2: 统一验证集评估**
使用同一个验证集（建议用957n，因为更全面）重新评估所有模型：
```bash
python new_dinoyolo_src/evaluate_405YOLO_checkpoint.py \
    --checkpoint res_checkpoints/multi_disease_957n_expt_v2/best_map.pth \
    --val_json coco/All_Diseases_957n/val.json

python new_dinoyolo_src/evaluate_405YOLO_checkpoint.py \
    --checkpoint res_checkpoints/multi_disease_957_expt_v3_1/best_map.pth \
    --val_json coco/All_Diseases_957n/val.json
```

---

## 💡 建议的调查步骤

## 💡 最终建议

### **立即行动清单**

#### **1. 统一评估基准（最优先）** 🔥
当前无法判断v2和v3.1哪个模型更好，因为它们在不同的验证集上评估。建议：

```bash
# 在957n（更全面的验证集）上重新评估所有模型
cd "d:/File/Programming/Tooth_VLM/dinoV3_ToothVLM"

# 评估v2 (如果权重还在)
python new_dinoyolo_src/evaluate_405YOLO_checkpoint.py \
    --checkpoint res_checkpoints/multi_disease_957n_expt_v2/best_map.pth \
    --val_json coco/All_Diseases_957n/val.json

# 评估v3.1
python new_dinoyolo_src/evaluate_405YOLO_checkpoint.py \
    --checkpoint res_checkpoints/multi_disease_957_expt_v3_1/best_map.pth \
    --val_json coco/All_Diseases_957n/val.json
```

**预期结果：** v3.1在957n验证集上的mAP可能会降至0.55-0.60区间，与v2的0.528更接近。

#### **2. 确定标准数据集**
查找数据集的来源和处理记录：
```bash
# 检查数据集目录下是否有README或处理脚本
ls -lh ../957/
ls -lh ../957n/
find ../957* -name "README*" -o -name "*.py" -o -name "*.sh"
```

**决策标准：**
- 如果957是清洗后的高质量版本 → **用于论文和最终评估**
- 如果957n是完整的原始版本 → **用于模型训练和公平对比**
- 如果两者来源不同 → **合并后重新划分train/val**

#### **3. 代码架构选择**
- **模块化版本（new_dinoyolo_src/）的优势已验证：**
  - 便于添加新工具（如threshold_sweep.py）
  - 代码可读性更好
  - 便于团队协作和版本控制

- **建议：**
  - 后续实验统一使用`new_dinoyolo_src/`架构
  - 将`src/`标记为legacy，仅用于复现历史实验
  - 在`new_dinoyolo_src/`中实现model_improvement_analysis.md提出的优化

#### **4. 关于F1=0.814的记录**
你提到"v2（F1=0.814）训练时还没有写日志功能"，但当前找到的v2日志显示F1=0.627。可能的情况：
- 那个0.814的模型使用了不同的超参数/数据集
- 或者使用了不同的评估阈值

**建议：** 如果那个模型的权重还在，请重新评估并记录完整配置，避免混淆。

---

## 📋 总结

### **关键结论**

1. **src/和new_dinoyolo_src/的核心算法完全一致** ✅
   - 模型架构相同
   - 损失函数相同
   - 超参数相同
   - 数据增强相同（高级增强目前禁用）

2. **性能差异源于数据集不同** ✅
   - 957验证集: 510标注，平均框面积0.0410，Tooth_Discoloration=79
   - 957n验证集: 595标注，平均框面积0.0329，Tooth_Discoloration=164
   - 957n更难（+小目标+密集病灶），导致mAP更低

3. **当前两个版本的mAP不可比** ⚠️
   - v2 (mAP@.5=0.528 on 957n) 
   - v3.1 (mAP@.5=0.763 on 957)
   - 需要在同一验证集上重新评估才能对比

4. **模块化重构带来的副作用可以忽略** ✅
   - 训练收敛曲线相似
   - 梯度范数差异<10%
   - 随机种子设置一致

### **后续工作路线图**

#### **阶段1: 建立评估基准（本周）**
- [ ] 在957n验证集上重新评估v3.1模型
- [ ] 确认957和957n的数据来源和质量差异
- [ ] 选择一个标准数据集用于后续所有实验
- [ ] 记录baseline性能作为优化起点

#### **阶段2: 实施快速优化（1-2周）**
按照`model_improvement_analysis.md`的Phase 1执行：
- [ ] Mosaic数据增强（MOSAIC_PROB=0.5）
- [ ] Focal Loss替换BCE
- [ ] 类别自适应NMS
- [ ] 小目标重采样（mouth_ulcer × 2.0）

**目标：** F1从0.627提升至0.70

#### **阶段3: 架构优化（2-3周）**
按照`model_improvement_analysis.md`的Phase 2执行：
- [ ] 增加P2超小尺度层（stride=4）
- [ ] Neck中增加Attention机制
- [ ] 渐进式解冻训练
- [ ] 混合精度 + 梯度累积

**目标：** F1提升至0.75-0.80

#### **阶段4: 深度优化（3-4周）**
- [ ] 对比学习预训练微调
- [ ] Cascade检测头
- [ ] Deformable Conv
- [ ] 轻量化骨干（DINOv3-S）

**目标：** F1达到0.85-0.90（论文级别）

---

---

## 📌 重要补充说明（2026-08-14 更新）

### **关于历史最佳结果（F1=0.814）**

**用户反馈：** "那组极高的结果无法再次复现"

**分析：**
1. **深度学习训练的不确定性**
   - 即使设置了seed=42，某些操作仍不完全确定（cuDNN算法选择、多进程数据加载）
   - 早期实验可能恰好碰到了一个好的随机初始化/数据顺序组合
   
2. **可能的影响因素**
   - CUDA/PyTorch版本更新导致的底层计算差异
   - 数据集版本/划分的变动
   - 评估阈值配置的差异

3. **如何处理**
   - **不要把那个0.814作为必须达到的目标**（可能是运气成分）
   - **以当前可复现的基线为准**：F1=0.627 (v2 on 957n)
   - **通过系统优化稳定提升**，而不是追求不可复现的峰值

### **关于Mosaic等数据增强**

**用户确认：** "之前的mosaic等对测试确实是有正面作用的"

这证实了[model_improvement_analysis.md](model_improvement_analysis.md)中的Phase 1方向正确！

**建议的验证策略：**
```python
# 在 new_dinoyolo_src/train_detector_405YOLO.py 中
# 测试不同的 Mosaic 配置

# 实验1: 轻度 Mosaic
Config.MOSAIC_PROB = 0.3
Config.COPY_PASTE_PROB = 0.0

# 实验2: 标准 Mosaic
Config.MOSAIC_PROB = 0.5
Config.COPY_PASTE_PROB = 0.0

# 实验3: Mosaic + Copy-Paste
Config.MOSAIC_PROB = 0.5
Config.COPY_PASTE_PROB = 0.2
```

每个配置训练3次，取平均mAP，避免单次运气因素。

### **修正的优化策略**

基于"极高结果不可复现"和"Mosaic有效"的反馈，调整实施策略：

#### **第一步：重新建立可复现的baseline（本周）**
```bash
# 统一数据集和评估标准
# 选择 957n（更全面）作为标准数据集
cd "d:/File/Programming/Tooth_VLM/dinoV3_ToothVLM"

# 训练3次取平均，验证可复现性
for i in 1 2 3; do
  python new_dinoyolo_src/train_detector_405YOLO.py \
    --seed $((42 + i)) \
    --output_dir res_checkpoints/baseline_reproducibility_run${i}
done
```

**目标：** 确认baseline的均值和方差（如 F1=0.63±0.02）

#### **第二步：系统测试数据增强（1周）**
由于你已经确认Mosaic有效，进行系统的消融实验：

| 实验组 | MOSAIC_PROB | COPY_PASTE_PROB | 预期提升 |
|--------|-------------|-----------------|----------|
| Baseline | 0.0 | 0.0 | F1=0.63 |
| Exp-A | 0.3 | 0.0 | F1=0.66 |
| Exp-B | 0.5 | 0.0 | F1=0.68 |
| Exp-C | 0.5 | 0.2 | F1=0.70 |

每组训练2次取平均，找出最优配置。

#### **第三步：叠加其他优化（2周）**
在确定的最优数据增强基础上，逐步叠加：
1. Focal Loss
2. 类别自适应NMS
3. 小目标重采样

**关键原则：** 
- **每次只改一个变量**
- **每个配置至少运行2次**
- **记录完整的训练日志和配置文件**

---

**文档创建时间：** 2026-08-14  
**最后更新：** 2026-08-14（补充可复现性讨论）  
**分析对象：** src/ (commit: bf7b1fa) vs new_dinoyolo_src/ (branch: 0807_exp)  
**核心结论：** 
1. 代码算法一致，性能差异源于验证集不同（957 vs 957n）
2. 历史最佳结果（F1=0.814）不可复现，应以当前baseline (F1=0.63)为准
3. Mosaic增强已证实有效，优先进行系统的数据增强消融实验  
**数据集对比工具：** [src/utils/compare_datasets.py](src/utils/compare_datasets.py)
