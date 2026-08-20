import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from new_dinoyolo_src.tools.tooth_detector_tool import detect_tooth_diseases
from new_dinoyolo_src.tools.function_definitions import (
    AVAILABLE_TOOLS,
    SYSTEM_PROMPT_WITH_TOOLS
)


def execute_function_call(function_name: str, arguments: dict) -> dict:
    """执行工具函数"""
    if function_name == "detect_tooth_diseases":
        try:
            result = detect_tooth_diseases(**arguments)
            return result
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e)
            }
    else:
        return {
            "status": "error",
            "error_message": f"未知的函数: {function_name}"
        }


def chat_with_lm_studio(image_path: str = "D:\\File\\Programming\\Tooth_VLM\\dinoV3_ToothVLM\\testIMG\\4.jpg"):
    """
    使用 LM Studio 进行完整的对话和工具调用

    Args:
        image_path: 要检测的图片路径
    """
    print("\n" + "=" * 70)
    print("🦷 口腔健康诊断助手 - LM Studio 版本")
    print("=" * 70)

    # 检查是否安装了 openai 库
    try:
        from openai import OpenAI
    except ImportError:
        print("\n 错误: 未安装 openai 库")
        print("   安装命令: pip install openai")
        return

    # 连接到 LM Studio
    print("\n 连接到 LM Studio...")
    client = OpenAI(
        base_url="http://localhost:1234/v1",  # LM Studio 默认地址
        api_key="lm-studio"  # 随便填，LM Studio 不验证
    )

    # 测试连接
    try:
        models = client.models.list()
        model_name = models.data[0].id if models.data else "qwen3-coder-30b"
        print(f" 已连接到 LM Studio")
        print(f"   使用模型: {model_name}\n")
    except Exception as e:
        print(f"\n 无法连接到 LM Studio: {e}")
        print("\n 请确保:")
        print("   1. LM Studio 已启动")
        print("   2. 已加载模型")
        print("   3. 已点击 'Start Server' 按钮")
        print("   4. 服务器运行在 http://localhost:1234\n")
        return

    # 构建对话
    user_message = f"你好，我有一张口腔图片，能帮我诊断一下吗？图片路径是：{image_path}"

    print("=" * 70)
    print(f" 用户: {user_message}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOLS},
        {"role": "user", "content": user_message}
    ]

    # 第一轮：让 LLM 决定是否调用工具
    print(" 助手: [思考中...]\n")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
            temperature=0.7,
        )

        assistant_message = response.choices[0].message

        # 检查是否有工具调用
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f" [LLM 决定调用工具] {function_name}")
            print(f"   参数: {json.dumps(arguments, ensure_ascii=False, indent=2)}\n")

            # 执行工具
            print(" 正在执行检测（首次加载模型需要 5-10 秒）...\n")
            result = execute_function_call(function_name, arguments)

            if result['status'] == 'success':
                print(" 检测完成！")
                print(f"   检测到: {result['detections_count']} 处异常")
                print(f"   标注图片: {result['annotated_image_path']}\n")
            else:
                print(f" 检测失败: {result.get('error_message', '未知错误')}\n")

            # 将工具调用和结果添加到对话历史
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }]
            })

            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "tool_call_id": tool_call.id
            })

            # 第二轮：让 LLM 基于工具结果生成最终回复
            print(" 助手: [根据检测结果生成回复...]\n")

            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
            )

            final_message = final_response.choices[0].message.content

            print("=" * 70)
            print("助手:\n")
            print(final_message)
            print("\n" + "=" * 70)

            # 显示诊断报告
            if result['status'] == 'success':
                print("\n 详细诊断报告:")
                print("=" * 70)
                print(result['diagnosis_report'])
                print("=" * 70)
                print(f"\n 标注图片已保存: {result['annotated_image_path']}")
                print("   可以打开查看具体的病灶位置\n")

        else:
            # LLM 没有调用工具，直接回复
            print(" 助手:\n")
            print(assistant_message.content)
            print("\n 提示: LLM 没有调用检测工具，可能是:")
            print("   1. LM Studio 版本较老，不支持 Function Calling")
            print("   2. 模型没有理解需要调用工具")
            print("   3. 可以尝试更明确的提示词\n")

    except Exception as e:
        print(f"\n 调用 LLM 时出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n 可能的原因:")
        print("   1. LM Studio 版本不支持 Function Calling")
        print("   2. 网络连接问题")
        print("   3. 模型加载失败\n")


def test_lm_studio_connection():
    """测试 LM Studio 连接"""
    print("\n" + "=" * 70)
    print(" 测试 LM Studio 连接")
    print("=" * 70 + "\n")

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="test"
        )

        # 获取模型列表
        models = client.models.list()

        if models.data:
            print(" 连接成功！")
            print("\n可用的模型:")
            for model in models.data:
                print(f"   - {model.id}")
            print("\n LM Studio 已就绪，可以开始对话！")
        else:
            print(" 连接成功，但没有加载模型")
            print("   请在 LM Studio 中加载一个模型")

        return True

    except ImportError:
        print("未安装 openai 库")
        print("   安装命令: pip install openai")
        return False

    except Exception as e:
        print(f"无法连接到 LM Studio: {e}")
        print("\n 请确保:")
        print("   1. LM Studio 已启动")
        print("   2. 已加载模型（如 qwen3-coder-30b）")
        print("   3. 已点击 'Start Server' 按钮")
        print("   4. 服务器运行在 http://localhost:1234")
        return False


def main():
    """主函数"""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🦷 口腔健康诊断 - LM Studio 集成" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")

    # 测试连接
    if not test_lm_studio_connection():
        return

    # 检查测试图片
    image_path = "D:\\File\\Programming\\Tooth_VLM\\dinoV3_ToothVLM\\testIMG\\4.jpg"
    if not Path(image_path).exists():
        print(f"\n 测试图片不存在: {image_path}")
        print("\n 请准备一张口腔图片:")
        print("   1. 创建目录: mkdir test_images")
        print("   2. 复制图片: cp your_image.jpg test_images/sample_tooth.jpg")
        print("\n或者修改代码中的 image_path 变量\n")

        # 询问是否继续（使用其他图片）
        custom_path = input("输入图片路径（留空退出）: ").strip()
        if custom_path and Path(custom_path).exists():
            image_path = custom_path
        else:
            return

    # 运行对话
    chat_with_lm_studio(image_path)

    print("\n" + "=" * 70)
    print("对话结束")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
