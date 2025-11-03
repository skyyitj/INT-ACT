echo "🔧 INT-ACT 模型文件一键修复"
echo "=========================================="
echo ""
echo "检测到错误: SafetensorError - 模型文件损坏"
echo "将删除损坏的模型并重新下载"
echo ""

MODEL_DIR="models/INTACT-pi0-finetune-bridge"

# 检查目录是否存在
if [ -d "$MODEL_DIR" ]; then
    echo "📦 找到模型目录: $MODEL_DIR"
    echo ""

    # 询问是否继续
    read -p "是否删除并重新下载? (y/N): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🗑️  删除损坏的模型..."
        rm -rf "$MODEL_DIR"
        echo "✅ 删除完成"
        echo ""

        echo "📥 重新下载模型..."
        echo "⏳ 这可能需要几分钟，请耐心等待..."
        echo ""

        # 使用 huggingface-cli 或 git 下载
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download juexzz/INTACT-pi0-finetune-bridge \
                --local-dir "$MODEL_DIR" \
                --local-dir-use-symlinks False
        else
            echo "💡 使用 git 下载..."
            git clone https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge "$MODEL_DIR"
        fi

        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 模型下载完成！"
            echo ""
            echo "📋 现在可以运行评估了:"
            echo "  python run_pi0_simpler_local.py --model-path ./$MODEL_DIR"
        else
            echo ""
            echo "❌ 下载失败，请检查网络连接"
            echo ""
            echo "💡 手动下载方法:"
            echo "  git clone https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge $MODEL_DIR"
        fi
    else
        echo "❌ 操作已取消"
    fi
else
    echo "❌ 模型目录不存在: $MODEL_DIR"
    echo ""
    read -p "是否下载模型? (Y/n): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo "📥 下载模型..."
        mkdir -p "models"

        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download juexzz/INTACT-pi0-finetune-bridge \
                --local-dir "$MODEL_DIR" \
                --local-dir-use-symlinks False
        else
            git clone https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge "$MODEL_DIR"
        fi

        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 模型下载完成！"
        else
            echo ""
            echo "❌ 下载失败"
        fi
    fi
fi

echo ""
echo "=========================================="

