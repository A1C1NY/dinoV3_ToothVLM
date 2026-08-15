# Cascade Stage2 Training Collapse Fix

## Problem Summary

At epoch 21 (transition from stage1 to stage2), catastrophic training failure occurred:
- **Loss**: 52.37 → 143.80 (3x increase)
- **mAP**: 0.223 → 0.048 (collapsed)
- **Gradient norm**: 62 → 627 (10x spike)
- **Predictions**: 528 → 101 boxes (detection nearly stopped)

## Root Causes

1. **Abrupt architecture change**: Stage2 parameters were randomly initialized (even though copied from stage1), while stage1 had converged for 20 epochs
2. **Loss discontinuity**: Stage2 produces different predictions due to refined features passing through different detection layers
3. **Gradient mismatch**: Stage2's 0.5× learning rate wasn't enough when transitioning from frozen to active state

## Implemented Fixes

### Fix 1: Gradual Warmup Transition (3 epochs)
**File**: [yolov10_dinov3.py](new_dinoyolo_src/model/yolov10_dinov3.py:102-193)

Added 3-epoch warmup period where stage1 and stage2 outputs are **blended**:
- Epoch 21: 0% stage2, 100% stage1
- Epoch 22: 50% stage2, 50% stage1  
- Epoch 23: 100% stage2, 0% stage1

```python
@property
def warmup_alpha(self):
    """Blend factor: 0.0 at warmup start, 1.0 at warmup end"""
    if not self.in_warmup:
        return 1.0 if self.stage2_enabled else 0.0
    progress = (self.current_epoch - self.stage1_epochs) / self.warmup_epochs
    return progress

def forward(self, features):
    if not self.stage2_enabled:
        return self.stage1(features)
    
    stage1_out = self.stage1(features)
    refined = [feature + refine(feature) for feature, refine in zip(features, self.stage2_refine)]
    stage2_out = self.stage2(refined)
    
    if self.in_warmup:
        alpha = self.warmup_alpha
        blended = [(1 - alpha) * s1 + alpha * s2 for s1, s2 in zip(stage1_out, stage2_out)]
        return blended
    
    return stage2_out
```

### Fix 2: Lower Initial Learning Rate for Stage2
**File**: [train_detector_405YOLO.py](new_dinoyolo_src/train_detector_405YOLO.py:263-267)

Reduced stage2 learning rate from **0.5× → 0.1×** the head LR:
```python
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": Config.BACKBONE_LR},      # 0.0001
    {"params": head_params, "lr": Config.LR},                   # 0.001
    {"params": stage2_params, "lr": Config.LR * 0.1},           # 0.0001 (was 0.0005)
], weight_decay=1e-4)
```

### Fix 3: Gradient Scaling During Warmup
**File**: [train_detector_405YOLO.py](new_dinoyolo_src/train_detector_405YOLO.py:290-299)

Scale stage2 gradients during warmup to prevent them from dominating the loss:
```python
# After backward, before optimizer.step()
if model.detect_head.in_warmup:
    scale = model.detect_head.warmup_alpha * 0.5  # 0.0 → 0.5 during warmup
    for param in stage2_params:
        if param.grad is not None:
            param.grad.mul_(scale)
```

### Fix 4: Training Stage Logging
**File**: [train_detector_405YOLO.py](new_dinoyolo_src/train_detector_405YOLO.py:304-315)

Added cascade stage status to training logs:
```
Epoch 20: loss=52.37, ... [Cascade Stage1]
Epoch 21: loss=X, ... [Cascade Warmup: 0.33]
Epoch 22: loss=X, ... [Cascade Warmup: 0.67]
Epoch 23: loss=X, ... [Cascade Warmup: 1.00]
Epoch 24: loss=X, ... [Cascade Stage2]
```

## Expected Behavior After Fix

### Training Timeline
- **Epochs 1-20**: Stage1 only, stage2 frozen
- **Epochs 21-23**: Warmup blending, gradual transition
- **Epochs 24-70**: Stage2 fully active

### Expected Metrics During Transition
- **Epoch 21**: Loss should stay close to epoch 20 (~52), mAP ~0.22
- **Epoch 22**: Small loss increase acceptable (~60-70), mAP ~0.20-0.22
- **Epoch 23**: Loss stabilizing, mAP recovering to ~0.22+
- **Epoch 24+**: Stage2 should improve mAP@0.75 gradually

## Configuration

Two new parameters added to [Config](new_dinoyolo_src/train_detector_405YOLO.py:92-94):
```python
CASCADE_STAGE1_EPOCHS = 20      # When to start stage2
CASCADE_WARMUP_EPOCHS = 3       # How long to blend
```

## Why This Works

1. **Smooth transition**: Blending prevents sudden prediction changes that spike the loss
2. **Conservative gradients**: Lower LR + gradient scaling keeps stage2 updates small initially
3. **Preserved stage1**: During warmup, stage1 still contributes to predictions, maintaining detection capability
4. **Natural convergence**: By epoch 24, stage2 has had 3 epochs of gentle adaptation before taking full control

## Monitoring

Watch for these signs of successful transition:
- ✅ Loss stays under 80 during warmup (epochs 21-23)
- ✅ Predictions stay above 300 boxes per validation
- ✅ Gradient norm stays under 200
- ✅ mAP doesn't drop below 0.18

If problems persist, increase `CASCADE_WARMUP_EPOCHS` to 5.
