# 本地环境评估指南 (Conda版本)

本文档说明如何在使用conda环境管理的本地服务器上运行pi0模型的评估脚本。

## 评估脚本位置

适配本地conda环境的评估脚本位于：
```bash
slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh
```

## 使用前准备

### 1. 手动激活conda环境

**在运行脚本前，必须先手动激活conda环境：**

```bash
conda activate your_env_name  # 替换为您的实际环境名
```

脚本会自动检查是否已激活conda环境，如果未激活会提示并退出。

**注意：**
- 如果服务器端和客户端使用相同的环境，激活一个环境即可
- 如果需要不同的环境，请参考下面的"高级配置"部分

### 2. 配置评估参数

#### 修改配置文件（第16行）
```bash
CONFIG_NAMES=("pi0_finetune_bridge_ev.yaml")
```

#### 修改随机种子（第17行）
```bash
SEEDS=(42 7 314)
```
- 脚本会按顺序使用每个种子进行评估
- 可以根据需要增加或减少种子数量

#### 修改批处理大小（第82行）
```bash
BATCH_SIZE=4
```
- 同时运行的服务器-客户端对数量
- **80GB GPU建议值：4**
- **40GB GPU建议值：2**
- **更小的GPU：1**
- 根据您的GPU显存调整此值

## 运行评估

### 方式1: 直接运行
```bash
# 1. 先激活conda环境
conda activate your_env_name

# 2. 运行评估脚本
cd /home/lishuang/intact
./slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh
```

### 方式2: 后台运行（推荐用于长时间评估）
```bash
# 1. 先激活conda环境
conda activate your_env_name

# 2. 后台运行
cd /home/lishuang/intact
nohup ./slurms/eval_scripts/simpler/ev_pi0_bridge_simpler_local.sh > eval_output.log 2>&1 &
```

查看运行日志：
```bash
tail -f eval_output.log
```

## 评估流程说明

脚本会执行以下步骤：

1. **读取配置**：从YAML配置文件中读取checkpoint步数列表
2. **遍历种子**：按顺序使用每个随机种子
3. **批量评估**：将checkpoint列表分批，每批同时评估BATCH_SIZE个checkpoint
4. **服务器-客户端架构**：
   - 服务器端：加载模型并提供推理服务
   - 客户端：运行SimplerEnv仿真并请求模型预测
5. **自动清理**：每批评估完成后自动终止服务器进程

## 评估结果

评估结果会保存在配置文件中指定的路径，通常为：
```
log/train/pi0_finetune_bridge/[训练时间戳]/eval_results/
```

## 常见问题

### Q1: 如何查看我的conda环境列表？
```bash
conda env list
```

### Q2: 如何检查conda环境中的包？
```bash
conda list -n intact
conda list -n simpler
```

### Q3: 如何减少GPU内存占用？
- 将`BATCH_SIZE`设置为更小的值（如1或2）
- 在配置文件中减少`n_eval_episode`的数量

### Q4: 评估时间太长怎么办？
- 减少`SEEDS`的数量（例如只用一个种子：`SEEDS=(42)`）
- 减少配置文件中的任务列表
- 减少`eval_cfg.pretrained_model_gradient_step_cnt`中的步数

### Q5: 端口冲突怎么办？
脚本会自动查找可用端口（10000-65500范围内），通常不会有冲突。如果仍有问题，可以检查：
```bash
ss -tuln | grep :XXXX  # 检查特定端口
```

## 与原始SLURM脚本的区别

相比原始的SLURM集群脚本（`ev_pi0_bridge_simpler.sh`），本地版本：

1. ✅ 移除了所有SLURM指令（#SBATCH）
2. ✅ 移除了Singularity容器调用
3. ✅ 改用conda环境管理
4. ✅ 添加了conda初始化
5. ✅ 简化了环境激活逻辑
6. ✅ 保留了核心的并行评估逻辑

## 高级配置

### 如果服务器和客户端需要不同的conda环境

如果您的服务器端和客户端需要不同的环境（如LeRobot在一个环境，SimplerEnv在另一个环境），可以修改脚本第124-135行：

```bash
# 原始代码（使用当前激活的环境）
python src/agent/run.py \
    --config_path config/experiment/simpler/${CONFIG_NAME} \
    ...

# 修改为使用特定环境（客户端）
conda run -n simpler_env --no-capture-output python src/agent/run.py \
    --config_path config/experiment/simpler/${CONFIG_NAME} \
    ...
```

### 单独评估某个checkpoint

如果只想评估特定的checkpoint，可以直接运行：

```bash
# 激活环境
conda activate your_env_name

# 切换到项目目录
cd /home/lishuang/intact

# 启动服务器
python src/agent/run.py \
    --config_path config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --seed 42 \
    --use_bf16 False \
    --eval_cfg.port 12345 \
    --eval_cfg.pretrained_model_gradient_step_cnt="[15130]" \
    --use_wandb False \
    --eval_cfg.role server &

# 等待2秒后启动客户端
sleep 2

python src/agent/run.py \
    --config_path config/experiment/simpler/pi0_finetune_bridge_ev.yaml \
    --seed 42 \
    --use_bf16 False \
    --eval_cfg.port 12345 \
    --eval_cfg.pretrained_model_gradient_step_cnt="[15130]" \
    --use_wandb False \
    --eval_cfg.role client
```

### 修改评估任务列表

编辑配置文件 `config/experiment/simpler/pi0_finetune_bridge_ev.yaml` 中的 `task_list` 来选择要评估的任务。

## 技术支持

如有问题，请参考：
- 主文档：`doc/evaluation.md`
- 原始SLURM脚本：`slurms/eval_scripts/simpler/ev_pi0_bridge_simpler.sh`
- 入口脚本：`src/agent/run.py`

