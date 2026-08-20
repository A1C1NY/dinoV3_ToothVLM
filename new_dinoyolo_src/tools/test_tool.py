"""
简单测试脚本：验证工具是否正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_tool_import():
    """测试工具导入"""
    print("🧪 测试 1: 检查工具导入...")
    try:
        from new_dinoyolo_src.tools.tooth_detector_tool import (
            SimpleToothDetector,
            detect_tooth_diseases
        )
        from new_dinoyolo_src.tools.function_definitions import (
            AVAILABLE_TOOLS,
            SYSTEM_PROMPT_WITH_TOOLS
        )
        print("   ✅ 所有模块导入成功\n")
        return True
    except Exception as e:
        print(f"   ❌ 导入失败: {e}\n")
        return False


def test_checkpoint_exists():
    """测试权重文件是否存在"""
    print("🧪 测试 2: 检查模型权重文件...")
    checkpoint_path = Path(__file__).resolve().parents[2] / "res_checkpoints" / "multi_disease_Sonata_expt_v3_1" / "best_map.pth"

    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ 权重文件存在: {checkpoint_path}")
        print(f"   📦 文件大小: {size_mb:.2f} MB\n")
        return True
    else:
        print(f"   ❌ 权重文件不存在: {checkpoint_path}")
        print(f"   💡 请确保训练完成并保存了模型权重\n")
        return False


def test_function_definition():
    """测试 Function Calling 定义"""
    print("🧪 测试 3: 检查 Function Calling 定义...")
    try:
        from new_dinoyolo_src.tools.function_definitions import TOOTH_DETECTION_FUNCTION

        assert TOOTH_DETECTION_FUNCTION["type"] == "function"
        assert "function" in TOOTH_DETECTION_FUNCTION
        assert TOOTH_DETECTION_FUNCTION["function"]["name"] == "detect_tooth_diseases"
        assert "parameters" in TOOTH_DETECTION_FUNCTION["function"]

        print("   ✅ Function Calling 定义格式正确")
        print(f"   📋 工具名称: {TOOTH_DETECTION_FUNCTION['function']['name']}")
        print(f"   📝 描述: {TOOTH_DETECTION_FUNCTION['function']['description'][:60]}...\n")
        return True
    except Exception as e:
        print(f"   ❌ 定义检查失败: {e}\n")
        return False


def test_model_loading():
    """测试模型加载"""
    print("🧪 测试 4: 尝试加载模型...")
    try:
        from new_dinoyolo_src.tools.tooth_detector_tool import SimpleToothDetector

        checkpoint_path = Path(__file__).resolve().parents[2] / "res_checkpoints" / "multi_disease_Sonata_expt_v3_1" / "best_map.pth"

        if not checkpoint_path.exists():
            print("   ⏭️  跳过（权重文件不存在）\n")
            return True

        print("   ⏳ 正在加载模型（可能需要几秒钟）...")
        detector = SimpleToothDetector(checkpoint_path, device="cpu")

        print("   ✅ 模型加载成功")
        print(f"   🖥️  运行设备: {detector.device}")
        print(f"   🎯 支持类别: {len(detector.CATEGORIES)} 种疾病\n")
        return True
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_with_sample_image():
    """使用示例图片测试完整流程"""
    print("🧪 测试 5: 使用示例图片测试...")

    # 查找测试图片
    project_root = Path(__file__).resolve().parents[2]
    possible_image_paths = [
        project_root / "test_images" / "sample_tooth.jpg",
        project_root / "../Sonata/image",  # 实际数据集目录
    ]

    sample_image = None
    for path in possible_image_paths:
        if path.is_file():
            sample_image = path
            break
        elif path.is_dir():
            # 查找目录中的第一张图片
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                images = list(path.glob(ext))
                if images:
                    sample_image = images[0]
                    break
            if sample_image:
                break

    if sample_image is None:
        print("   ⏭️  跳过（未找到测试图片）")
        print("   💡 提示: 将测试图片放在 test_images/sample_tooth.jpg\n")
        return True

    try:
        from new_dinoyolo_src.tools.tooth_detector_tool import detect_tooth_diseases

        print(f"   📷 使用图片: {sample_image}")
        print("   ⏳ 正在推理...")

        result = detect_tooth_diseases(
            image_path=str(sample_image),
            confidence_threshold=0.3
        )

        if result['status'] == 'success':
            print("   ✅ 推理成功")
            print(f"   🔍 检测到: {result['detections_count']} 处异常")
            print(f"   💾 标注图片: {result['annotated_image_path']}")
            print(f"\n   📋 诊断报告:")
            print("   " + "\n   ".join(result['diagnosis_report'].split('\n')[:5]))
            print("   ...\n")
            return True
        else:
            print(f"   ❌ 推理失败: {result.get('error_message', '未知错误')}\n")
            return False

    except Exception as e:
        print(f"   ❌ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("🦷 牙齿疾病检测工具 - 测试套件")
    print("=" * 70)
    print()

    results = []

    # 运行测试
    results.append(("导入测试", test_tool_import()))
    results.append(("权重文件检查", test_checkpoint_exists()))
    results.append(("Function 定义检查", test_function_definition()))
    results.append(("模型加载测试", test_model_loading()))
    results.append(("完整流程测试", test_with_sample_image()))

    # 汇总结果
    print("=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}  {name}")

    print()
    print(f"总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！工具已就绪。")
        print("\n下一步:")
        print("  1. 运行示例: python new_dinoyolo_src/examples/chat_with_detection.py")
        print("  2. 查看文档: new_dinoyolo_src/tools/README.md")
        print("  3. 集成到你的 LLM 项目中")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")

    print("=" * 70)


if __name__ == "__main__":
    main()
