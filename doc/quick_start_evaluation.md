# 快速开始：使用HuggingFace预训练模型进行评估

## ✅ 当前配置状态

### 模型信息
- **模型位置**：`/home/lishuang/intact/models/INTACT-pi0-finetune-bridge/`
- **模型来源**：HuggingFace - `juexzz/INTACT-pi0-finetune-bridge`
- **模型大小**：6.1GB
- **最佳checkpoint步数**：7565 (第5个epoch)
- **预期性能**（SimplerEnv原始4个任务）：
  - carrot_on_plate: 36.1%
  - eggplant_in_basket: 81.9%
  - stack_cube: 26.4%
  - spoon_on_towel: 45.8%

### 配置文件
- **评估配置**：`config/experiment/simpler/pi0_finetune_bridge_ev.yaml`
- **模型路径**：`./models/INTACT-pi0-finetune-bridge`
- **评估步数**：`[7565]`（最佳checkpoint）
- **评估任务数**：65个任务（包括原始任务、干扰任务、泛化任务、语言变体等）
- **每个任务episode数**：24

## 🚀 运行评估

### 方式1：完整评估（所有种子和任务）

```bash
# 1. 激活conda环境
conda activate your_env_name

# 2. 切换到项目目录
cd /home/lishuang/intact

# 3. 运行评估脚本
./slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh
```

这将使用3个随机种子（42, 7, 314）评估所有65个任务。

### 方式2：后台运行（推荐）

```bash
conda activate your_env_name
cd /home/lishuang/intact
nohup ./slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh > eval_output.log 2>&1 &

# 查看日志
tail -f eval_output.log
```

### 方式3：快速测试（单个种子）

如果想先快速测试，可以修改评估脚本中的种子设置：

编辑 `slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh` 第14行：
```bash
# 从
SEEDS=(42 7 314)

# 改为（只用一个种子）
SEEDS=(42)
```

### 方式4：最小化测试（单个任务）

如果想最快速地验证环境是否正确，可以：

1. 编辑配置文件 `config/experiment/simpler/pi0_finetune_bridge_ev.yaml`
2. 临时只保留一个任务测试：

```yaml
task_list: [
    "widowx_carrot_on_plate",  # 只测试这一个任务
]
```

然后运行评估脚本。

## 📊 评估结果位置

评估完成后，结果会保存在：
```
./results/simpler/pi0_finetune_bridge_ev/
```

包含：
- 成功率统计
- 评估视频（如果recording: True）
- 详细的评估日志

## ⚙️ 调整GPU资源使用

如果您的GPU显存较小，可以调整批处理大小：

编辑 `slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh` 第82行：

```bash
BATCH_SIZE=4  # 80GB GPU推荐值
# BATCH_SIZE=2  # 40GB GPU推荐值
# BATCH_SIZE=1  # 24GB或更小GPU推荐值
```

## 🔍 预期运行时间

根据您的配置：
- **任务数**：65个任务
- **每个任务episode数**：24
- **随机种子数**：3
- **总评估次数**：65 × 24 × 3 = 4,680次

预计总时间：**数小时到一天**（取决于GPU性能和并行度）

## ⚠️ 注意事项

1. **确保conda环境已激活**
   - 脚本会检查 `$CONDA_DEFAULT_ENV` 变量
   - 如果未激活会提示并退出

2. **确保SimplerEnv依赖已安装**
   - ManiSkill2_real2sim
   - SimplerEnv环境配置
   - 相关的仿真资源

3. **检查路径配置**
   - 确保 `set_path.sh` 中的路径正确
   - 确保 `MS2_REAL2SIM_ASSET_DIR` 等环境变量已设置

4. **端口冲突**
   - 脚本会自动查找可用端口（10000-65500）
   - 如有问题可以手动检查：`ss -tuln | grep :端口号`

## 🐛 故障排查

### 问题1：找不到模型文件
```bash
# 检查模型文件是否存在
ls -lh ./models/INTACT-pi0-finetune-bridge/

# 应该看到：
# - config.json
# - model.safetensors (约6.1GB)
# - README.md
```

### 问题2：conda环境未激活
```bash
# 检查当前环境
echo $CONDA_DEFAULT_ENV

# 如果为空，激活环境
conda activate your_env_name
```

### 问题3：SimplerEnv相关错误
```bash
# 检查环境变量
echo $MS2_REAL2SIM_ASSET_DIR
echo $MS_ASSET_DIR

# 应该指向正确的资源路径
```

### 问题4：GPU内存不足
- 减少 `BATCH_SIZE` 到 1
- 减少 `n_eval_episode` 到更小的值（如12）

## 📚 更多信息

- 完整评估文档：`doc/evaluation.md`
- Conda环境配置：`doc/evaluation_local_conda.md`
- 模型信息：`models/INTACT-pi0-finetune-bridge/README.md`
- 项目主页：https://github.com/ai4ce/INT-ACT

## 🎯 下一步

1. ✅ 模型已下载并配置完成
2. ✅ 评估脚本已适配本地环境
3. ⏭️ 激活conda环境并运行评估

准备好了就可以开始评估了！祝您评估顺利！🚀

