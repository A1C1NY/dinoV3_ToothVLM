# 🦷 牙齿疾病检测工具 - LLM 集成指南

将训练好的牙齿疾病检测模型作为工具供大语言模型（Qwen、GLM 等）调用。

## 📁 文件结构

```
new_dinoyolo_src/
├── tools/
│   ├── tooth_detector_tool.py      # 核心检测工具
│   └── function_definitions.py     # Function Calling 定义
└── examples/
    └── chat_with_detection.py      # 完整对话示例
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision pillow
```

### 2. 独立使用检测工具（不需要 LLM）

```python
from new_dinoyolo_src.tools.tooth_detector_tool import detect_tooth_diseases

# 直接检测图片
result = detect_tooth_diseases(
    image_path="path/to/your/tooth_image.jpg",
    confidence_threshold=0.3
)

# 查看结果
print(result['diagnosis_report'])
print(f"标注图片保存在: {result['annotated_image_path']}")
```

### 3. 集成到 LLM（以 Qwen 为例）

```python
from openai import OpenAI
from new_dinoyolo_src.tools.function_definitions import AVAILABLE_TOOLS, SYSTEM_PROMPT_WITH_TOOLS
from new_dinoyolo_src.tools.tooth_detector_tool import detect_tooth_diseases

# 初始化客户端
client = OpenAI(
    api_key="your-api-key",
    base_url="http://your-qwen-endpoint/v1"
)

# 发送消息
response = client.chat.completions.create(
    model="qwen-7b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOLS},
        {"role": "user", "content": "帮我诊断这张口腔图片: tooth.jpg"}
    ],
    tools=AVAILABLE_TOOLS,
    tool_choice="auto"
)

# 如果模型调用工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    
    # 执行检测
    result = detect_tooth_diseases(**arguments)
    
    # 返回结果给 LLM 生成最终回复
    # ...
```

## 🎯 功能特性

### 支持的疾病类型
- ✅ **龋齿** (Caries)
- ✅ **牙结石** (Calculus) 
- ✅ **口腔溃疡** (Mouth Ulcer)
- ✅ **牙齿变色** (Tooth Discoloration)

### 输出内容
1. **标注图片**: 在原图上绘制检测框和标签
2. **诊断报告**: 包含疾病统计和健康建议
3. **结构化数据**: JSON 格式的检测结果

### 示例输出

```json
{
  "status": "success",
  "annotated_image_path": "path/to/tooth_image_detected.jpg",
  "diagnosis_report": "📋 **口腔健康诊断报告**\n\n检测到 **2** 处异常：\n\n• **龋齿** (检出 1 处，平均置信度: 92.3%)\n  💡 建议尽快就诊进行充填治疗...",
  "detections_count": 2
}
```

## 📖 详细用法

### 调整置信度阈值

```python
# 降低阈值：检测更多可疑区域（可能有误报）
result = detect_tooth_diseases("image.jpg", confidence_threshold=0.2)

# 提高阈值：只保留高置信度结果（可能漏检）
result = detect_tooth_diseases("image.jpg", confidence_threshold=0.5)
```

### 自定义输出路径

```python
from new_dinoyolo_src.tools.tooth_detector_tool import SimpleToothDetector

detector = SimpleToothDetector(
    checkpoint_path="res_checkpoints/multi_disease_Sonata_expt_v3_1/best_map.pth"
)

result = detector.process_image(
    image_path="input.jpg",
    output_dir="my_results/",
    confidence_threshold=0.3
)
```

## 🧪 运行示例

```bash
# 查看完整的对话示例
python new_dinoyolo_src/examples/chat_with_detection.py
```

这会展示：
1. 模拟用户与 LLM 的完整对话
2. LLM 如何调用检测工具
3. 如何将结果整合到回复中
4. 集成到实际项目的指南

## 🔧 技术细节

### 模型架构
- **Backbone**: DINOv3 (ViT-B/16)
- **Neck**: PAN/FPN
- **Head**: YOLOv10 Detection Head

### 推理流程
1. 图像预处理（Letterbox resize → 768×768）
2. DINOv3 特征提取
3. YOLOv10 检测
4. 后处理（NMS + 置信度过滤）
5. 坐标转换回原图
6. 可视化绘制

### 性能
- **推理速度**: ~200ms/张 (GPU)
- **内存占用**: ~2GB (模型 + 推理)
- **支持设备**: CUDA / CPU

## 📝 注意事项

1. **模型路径**: 默认使用 `res_checkpoints/multi_disease_Sonata_expt_v3_1/best_map.pth`
2. **图像格式**: 支持 JPG、PNG 等常见格式
3. **输入尺寸**: 自动 resize，建议原图不要过大（<4K）
4. **医疗免责**: 此工具仅供辅助参考，不能替代专业医生诊断

## 🔗 相关资源

- [DINOv3 论文](https://arxiv.org/abs/2304.07193)
- [YOLOv10 论文](https://arxiv.org/abs/2405.14458)
- [Qwen Function Calling 文档](https://qwen.readthedocs.io/)

## 💡 常见问题

**Q: 如何提高检测准确率？**
A: 调整 `confidence_threshold`，或者提供更清晰的口腔图片。

**Q: 支持视频检测吗？**
A: 目前只支持静态图片，可以通过逐帧提取实现视频检测。

**Q: 可以检测其他疾病吗？**
A: 需要重新标注数据并训练模型，当前只支持 4 种疾病。

**Q: 不使用 LLM 可以直接调用吗？**
A: 可以！直接调用 `detect_tooth_diseases()` 函数即可。

## 📄 许可证

本项目基于研究目的开发，仅供学习和演示使用。
