"""
Function Calling 工具定义
供 Qwen、GLM、GPT 等支持 Function Calling 的 LLM 使用
"""

# OpenAI Function Calling 格式定义
TOOTH_DETECTION_FUNCTION = {
    "type": "function",
    "function": {
        "name": "detect_tooth_diseases",
        "description": (
            "检测口腔 X 光片或照片中的牙齿疾病。"
            "可以识别龋齿(caries)、牙结石(calculus)、口腔溃疡(mouth_ulcer)、牙齿变色(tooth_discoloration)等疾病。"
            "返回标注了检测框的图片和详细的诊断报告。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "口腔图片的文件路径，支持 jpg、png 等常见格式"
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "检测置信度阈值，范围 0-1，默认 0.3。较低的值会检测出更多可疑区域，较高的值只保留高置信度结果",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["image_path"]
        }
    }
}

# 工具列表（可以包含多个工具）
AVAILABLE_TOOLS = [TOOTH_DETECTION_FUNCTION]


# 系统提示词模板
SYSTEM_PROMPT_WITH_TOOLS = """你是一个口腔健康助手，可以帮助用户分析口腔图片，诊断潜在的牙齿疾病。

你可以使用以下工具：
- detect_tooth_diseases: 检测口腔图片中的疾病（龋齿、牙结石、口腔溃疡、牙齿变色等）

使用指南：
1. 当用户提供口腔图片并要求诊断时，调用 detect_tooth_diseases 工具
2. 工具会返回标注图片路径和详细诊断报告
3. 你需要将诊断报告以清晰、友好的方式呈现给用户
4. 提醒用户这只是辅助诊断，不能替代专业医生的诊断

注意事项：
- 如果用户没有提供图片路径，礼貌地询问图片位置
- 建议用户在检测到问题时及时就医
- 避免给出过于确定的医疗建议
"""


# 使用示例
if __name__ == "__main__":
    import json

    print("=== Function Calling 工具定义 ===\n")
    print(json.dumps(TOOTH_DETECTION_FUNCTION, indent=2, ensure_ascii=False))

    print("\n=== 系统提示词 ===\n")
    print(SYSTEM_PROMPT_WITH_TOOLS)
