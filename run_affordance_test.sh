#!/bin/bash
# Affordance功能测试脚本

echo "=============================================="
echo "  Affordance功能测试套件"
echo "=============================================="
echo ""

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

echo "Python版本: $(python --version)"
echo ""

# 菜单
echo "请选择要运行的测试:"
echo "  1) 快速演示 (推荐首次运行)"
echo "  2) 可视化样式测试"
echo "  3) 动作序列测试"
echo "  4) 环境包装器测试"
echo "  5) 使用指南"
echo "  6) 运行所有测试"
echo "  0) 退出"
echo ""

read -p "请输入选项 (0-6): " choice

case $choice in
    1)
        echo ""
        echo "=== 运行快速演示 ==="
        python demo_affordance.py
        ;;
    2)
        echo ""
        echo "=== 运行可视化样式测试 ==="
        python get_pose_corrected_coordinates.py --affordance
        ;;
    3)
        echo ""
        echo "=== 运行动作序列测试 ==="
        python get_pose_corrected_coordinates.py --affordance-actions
        ;;
    4)
        echo ""
        echo "=== 运行环境包装器测试 ==="
        python affordance_wrapper.py
        ;;
    5)
        echo ""
        echo "=== 显示使用指南 ==="
        python affordance_usage_guide.py
        ;;
    6)
        echo ""
        echo "=== 运行所有测试 ==="
        echo ""

        echo "1/4: 快速演示"
        python demo_affordance.py
        echo ""

        echo "2/4: 可视化样式测试"
        python get_pose_corrected_coordinates.py --affordance
        echo ""

        echo "3/4: 动作序列测试"
        python get_pose_corrected_coordinates.py --affordance-actions
        echo ""

        echo "4/4: 环境包装器测试"
        python affordance_wrapper.py
        echo ""

        echo "✅ 所有测试完成！"
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "测试完成！"
echo ""
echo "查看生成的图像文件："
echo "  - demo_*.png"
echo "  - test_*.png"
echo "  - action_*.png"
echo "  - *_comparison.png"
echo ""
echo "阅读文档："
echo "  - README_affordance.md (完整文档)"
echo "  - AFFORDANCE_CHANGES_SUMMARY.md (修改总结)"
echo ""
echo "下一步："
echo "  1. 查看生成的图像"
echo "  2. 阅读 README_affordance.md"
echo "  3. 在训练代码中集成 AffordanceWrapper"
echo "=============================================="

