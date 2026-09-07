# gsm8k-grpo-trainer

使用 GRPO (Group Relative Policy Optimization) 算法在 GSM8K 数学推理数据集上微调 Qwen2.5 系列大语言模型的训练项目。

## 项目简介

本项目实现了基于 GRPO 的 RLHF（基于人类反馈的强化学习）训练流程，针对 GSM8K 小学数学推理任务进行对齐训练。核心特点：

- **GRPO 算法**：组内相对策略优化，无需训练额外的 Reward Model 和 Critic 网络
- **Qwen2.5 基座**：支持 Qwen2.5 系列模型（1.5B / 3B）
- **多维度奖励**：格式奖励 + 答案准确率奖励的组合奖励机制
- **TensorBoard 可视化**：训练指标、评估准确率、优秀轨迹实时记录

## 项目结构

```
grpo-project/
├── data_types.py      # Episode / MiniBatch 数据结构定义
├── grpo.py            # GRPO 核心算法（轨迹采集 + 策略更新）
├── gsm8k_task.py      # GSM8K 数据集加载 + 奖励函数实现
├── train.py           # 主训练入口
├── train-00000-of-00001.jsonl   # GSM8K 训练集
├── test-00000-of-00001.jsonl    # GSM8K 测试集
└── README.md
```

## 环境要求

- Python >= 3.10
- CUDA（推荐，CPU 训练极慢）
- 显存要求：
  - Qwen2.5-1.5B-Instruct (bfloat16)：约 4GB 显存
  - Qwen2.5-3B-Instruct (bfloat16)：约 8GB 显存

---

## 1. 环境管理（uv）

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 作为 Python 包管理器。

### 1.1 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后，执行 `source $HOME/.local/bin/env` 或重新登录 shell 使环境变量生效。

### 1.2 创建虚拟环境并安装依赖

在项目根目录下执行：

```bash
# 创建虚拟环境（使用 Python 3.11，兼容性最佳），环境名：gsm8k-grpo-trainer
uv venv --python 3.11 gsm8k-grpo-trainer

# 激活虚拟环境
source gsm8k-grpo-trainer/bin/activate

# 安装项目依赖
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.44.2 tokenizers==0.19.1
uv pip install modelscope==1.18.0
uv pip install numpy==1.26.4
uv pip install tensorboard==2.17.0
```

### 1.3 一键安装所有依赖

创建 `pyproject.toml` 后，指定虚拟环境目录直接同步：

```bash
uv sync --env gsm8k-grpo-trainer
```

---

## 2. 模型下载

本项目默认使用 [ModelScope](https://modelscope.cn/) 下载模型（国内网络速度更快）。

### 2.1 下载 Qwen2.5-1.5B-Instruct（项目默认配置，推荐入门）

```bash
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ./Qwen2.5-1.5B-Instruct
```

### 2.2 下载 Qwen2.5-3B-Instruct（效果更好，显存 ≥ 10GB 时使用）

```bash
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct
```

### 2.3 切换模型版本

修改 `train.py` 中的模型路径：

```python
# 默认配置（1.5B 版本）
pretrained_model_path = Path("./Qwen2.5-1.5B-Instruct/")

# 改为 3B 版本（显存足够时推荐，效果更好）
pretrained_model_path = Path("./Qwen2.5-3B-Instruct/")
```

---

## 3. 开始训练

### 3.1 启动训练

```bash
# 确保虚拟环境已激活
python train.py
```

### 3.2 超参数说明

在 `train.py` 中调整采样相关超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 256 | 每轮采样的总轨迹数 |
| `NUM_QUESTIONS_PER_BATCH` | 32 | 每轮采样的问题数量 |
| `NUM_ANSWERS_PER_QUESTION` | 8 | 每个问题采样的回答数（GRPO组大小） |

在 `train.py` 中调整优化器相关超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 1e-5 | 学习率 |
| `micro_batch_size` | 2 | 策略梯度微批次大小 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

### 3.3 查看训练日志

```bash
# 启动 TensorBoard（后台运行，端口 6006）
tensorboard --logdir ./logs --bind_all --port 6006 &
```

- **AutoDL 平台**：在实例的「自定义服务」中映射 `6006` 端口，然后通过平台提供的访问链接打开 TensorBoard 面板，查看训练曲线、评估准确率、满分样本等。

---

## 4. 奖励机制

总奖励公式（见 `gsm8k_task.py` 中 `reward_function` 函数）：

```
总奖励 = 0.1 * 格式奖励 + 答案准确性奖励
```

### 4.1 格式奖励（0.1 权重）

检查模型输出是否包含规范的 `<think>...</think><answer>...</answer>` 标签：

| 情况 | 格式奖励分数 |
|------|--------------|
| 完整匹配 `<think>...</think>\n<answer>...</answer>` | 1.0 |
| 只有 `<answer>...</answer>` 标签对 | 0.5 |
| 只有 `<think>...</think>` 标签对 | 0.1 |
| 无任何标签 | 0.0 |

### 4.2 答案准确性奖励（1.0 权重）

提取 `<answer>` 标签中的数值，与正确答案精确比较：

| 情况 | 答案奖励 |
|------|----------|
| 答案完全正确（支持整数/小数/千位分隔符） | 1.0 |
| 格式错误或答案错误 | 0.0 |

---

## 5. 检查点保存

- 每 100 步自动保存模型权重到 `./ckpt/ckpt_XXXXXX.pt`
- 每 10 步在测试集上评估一次准确率
- 满分样本会记录到 TensorBoard 的 Text 面板
