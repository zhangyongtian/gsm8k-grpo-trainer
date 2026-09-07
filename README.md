# gsm8k-grpo-trainer

使用 GRPO (Group Relative Policy Optimization) 算法在 GSM8K 数学推理数据集上微调 Qwen2.5 系列大语言模型的训练项目。

## 项目简介

本项目实现了基于 GRPO 的 RLHF（基于人类反馈的强化学习）训练流程，针对 GSM8K 小学数学推理任务进行对齐训练。核心特点：

- **GRPO 算法**：组内相对策略优化，无需训练额外的 Reward Model 和 Critic 网络
- **Qwen2.5 基座**：支持 Qwen2.5 系列模型（1.5B / 3B）
- **多维度奖励**：格式奖励 + 答案准确率奖励的组合奖励机制
- **TensorBoard 可视化**：训练指标、评估准确率、优秀轨迹实时记录
- **配置文件驱动**：所有超参数、路径均通过根目录 `.env` 配置，无需改代码
- **交互式测试**：训练完成后，直接用命令行给模型「出题」，看推理过程和答案

## 项目结构

```
gsm8k-grpo-trainer/
├── src/
│   └── gsm8k_grpo_trainer/       # Python 源码包
│       ├── __init__.py
│       ├── data_types.py         # Episode / MiniBatch 数据结构定义
│       ├── grpo.py               # GRPO 核心算法（轨迹采集 + 策略更新）
│       ├── gsm8k_task.py         # GSM8K 数据集加载 + 奖励函数实现
│       ├── train.py              # 训练入口（读 .env 配置）
│       └── inference.py          # 交互式出题推理（您手动一题一题测）
├── data/                         # GSM8K 数据集（Git 可追踪）
│   ├── train-00000-of-00001.jsonl
│   └── test-00000-of-00001.jsonl
├── logs/                         # TensorBoard 日志目录（Git 忽略）
├── ckpt/                         # 模型检查点目录（Git 忽略）
├── .env                          # 本地配置文件（不入库，由 .env.example 复制）
├── .env.example                  # 配置文件模板
├── .gitignore
├── pyproject.toml                # 项目配置 & 依赖声明
└── README.md
/root/gsm8k-grpo-trainer-venv/  # 虚拟环境（放在项目同级目录，避免套娃，Git 忽略）
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
source $HOME/.local/bin/env
```

> 说明：如果不执行 `source` 命令，也可以重新登录 shell 使环境变量生效。

### 1.2 创建虚拟环境并安装依赖

复制下面 4 条命令直接执行就行，不用管为什么拆分：

```bash
cd /root/gsm8k-grpo-trainer
# ① 创建虚拟环境
uv venv --python 3.11 /root/gsm8k-grpo-trainer-venv
# ② 激活
source /root/gsm8k-grpo-trainer-venv/bin/activate
# ③ 装 torch CUDA 版（直接给阿里云 cu121 上完整 wheel 文件 URL，不走 simple index 解析，100% 命中）
uv pip install "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.4.0+cu121-cp311-cp311-linux_x86_64.whl"
# ④ 装 pyproject.toml 里声明的所有其他依赖 + 注册 gsm8k-grpo-train / gsm8k-grpo-eval 命令
#    --no-upgrade：已经装好的 torch 2.4.0+cu121（CUDA 版）不要被 uv 想替换成 PyPI 上的 2.4.0（CPU 版）
uv sync --active --no-upgrade
```

> 说明：为什么 pyproject.toml 里写 `torch==2.4.0` 而不是 `==2.4.0+cu121`？
> - PyPI（以及清华镜像）上只有 `torch==2.4.0`（CPU 版），PyTorch 官方把带 CUDA 后缀的 `+cu121` 版本都放在自己的 `download.pytorch.org/whl/cu121` 独立镜像里，不在 PyPI 发布
> - 所以 pyproject.toml 里写 `==2.4.0` 让 uv sync 解析依赖时能在清华 PyPI 上正常「搜到」torch 这个包的主版本；而第 ③ 步我们已经先手动把 CUDA 版 `2.4.0+cu121` 装到 venv 里了，uv 会认为「已装的 `2.4.0+cu121` 满足 `==2.4.0` 约束」，就不会真的重新下载 CPU 版
> - 如果第 ③ 步没跑过、torch 还没装过，uv sync 会尝试去清华 PyPI 拿 CPU 版 `2.4.0`，这时候一定要中断，先执行第 ③ 步再跑 ④

> 镜像加速：
> - 第 ③ 步 torch → 阿里云 pytorch cu121 国内镜像（已验证有包，速度快）
> - 第 ④ 步其他包 → 清华 PyPI 国内镜像

> 重要提醒（今天踩过的坑）：
> - 如果 uv sync 输出类似 `- torch==2.4.0+cu121 ... + tqdm==x.x.x` 这种「减号 torch」的 diff 信息，然后您跑验证时真的报 `ModuleNotFoundError: No module named 'torch'` → 说明 uv 意外把 torch 卸掉了，**立刻重新执行第 ③ 步重装**就行，后面再跑 uv sync --no-upgrade 就不会再碰了

**✅ 装完后建议验证（确认 torch CUDA 版装对了）：**

```bash
python -c "import torch, transformers, modelscope, dotenv; print('torch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('其余依赖 OK')"
# 安装 modelscope
uv pip install modelscope

modelscope --version
which gsm8k-grpo-train
```

正常输出类似：
```
torch版本: 2.4.0+cu121
CUDA可用: True
CUDA版本: 12.1
其余依赖 OK
modelscope version 1.18.0
/root/gsm8k-grpo-trainer-venv/bin/gsm8k-grpo-train
```

### 1.3 虚拟环境常用命令

**激活虚拟环境：**
```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
```

**退出虚拟环境：**
```bash
deactivate
```

**删除虚拟环境（想重来时用）：**
```bash
rm -rf /root/gsm8k-grpo-trainer-venv
```

> 提示：执行任何项目命令（下载模型、训练、启动 TensorBoard、推理出题）之前，都需要先激活虚拟环境。

---

## 2. 配置文件（.env）

项目所有超参数、模型路径、推理用的 ckpt 路径，**全部通过根目录的 `.env` 文件配置**，无需修改代码。

首次使用先复制模板：

```bash
cd /root/gsm8k-grpo-trainer
cp .env.example .env
```

`.env` 配置项说明（直接在文件里改就行）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PRETRAINED_MODEL_PATH` | `Qwen2.5-1.5B-Instruct` | 基座模型目录（相对项目根目录，或绝对路径） |
| `CKPT_PATH` | 空 | **推理用**：训练后检查点路径，填 `ckpt/ckpt_001000.pt`；留空则对比「基座模型原始效果」 |
| `DEVICE` | `cuda` | 训练/推理设备：`cuda` / `cpu` |
| `DTYPE` | `bfloat16` | 权重精度：`bfloat16` / `float16` / `float32` |
| `SEED` | `1337` | 随机种子 |
| `BATCH_SIZE` | `256` | 每轮总轨迹数（= 问题数 × 每问回答数） |
| `NUM_QUESTIONS_PER_BATCH` | `32` | 每轮采样的问题数量 |
| `NUM_ANSWERS_PER_QUESTION` | `8` | 每个问题采样多少回答（GRPO 组大小） |
| `LR` | `1e-5` | 学习率 |
| `MICRO_BATCH_SIZE` | `2` | 策略梯度微批次大小 |
| `MAX_GRAD_NORM` | `1.0` | 梯度裁剪阈值 |
| `TEST_SIZE` | `128` | 训练时评估最多用多少条测试集 |
| `EVAL_EVERY` | `10` | 每多少步做一次测试集评估 |
| `SAVE_EVERY` | `100` | 每多少步保存一次 ckpt |

---

## 3. 模型下载

本项目默认使用 [ModelScope](https://modelscope.cn/) 下载模型（国内网络速度更快）。

> **前提说明**：请先完成上一节的 `uv sync` 步骤，modelscope CLI 工具已随依赖一起安装。确保虚拟环境已激活：
> ```bash
> cd /root/gsm8k-grpo-trainer
> source /root/gsm8k-grpo-trainer-venv/bin/activate
> modelscope --version
> ```

### 3.1 下载 Qwen2.5-1.5B-Instruct（项目默认配置，推荐入门）

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ./Qwen2.5-1.5B-Instruct
```

### 3.2 下载 Qwen2.5-3B-Instruct（效果更好，显存 ≥ 10GB 时使用）

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./Qwen2.5-3B-Instruct
```

### 3.3 切换模型版本

直接在 `.env` 文件里修改：

```dotenv
# 使用 1.5B
PRETRAINED_MODEL_PATH=Qwen2.5-1.5B-Instruct

# 使用 3B
# PRETRAINED_MODEL_PATH=Qwen2.5-3B-Instruct
```

---

## 4. 训练

### 4.1 启动训练

直接启动即可，**所有超参数都在 `.env` 里配置好了**：

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
# 两种方式任选其一
# 方式一：python -m 调用
python -m gsm8k_grpo_trainer.train
# 方式二：命令行入口
gsm8k-grpo-train
```

- 每 10 步会在测试集上评估一次准确率
- 每 100 步自动保存一次模型权重到 `ckpt/ckpt_XXXXXX.pt`（只存 model.state_dict，文件更小）

### 4.1.1 常见报错：CUDA OOM（显存不够用）

如果启动训练后报 `torch.OutOfMemoryError: CUDA out of memory`，按照下面**从上到下的顺序慢慢调小 `.env` 里的参数**，每次改完重新启动训练：

| 顺序 | 改什么 | 推荐值（24G 显卡） | 说明 |
|------|--------|------------------|------|
| ① | `NUM_ANSWERS_PER_QUESTION` | 从 8 → 4 → 2 | 每题生成的答案数，对显存影响最大（每多一个回答，就多一条完整生成轨迹）|
| ② | `NUM_QUESTIONS_PER_BATCH` | 从 32 → 8 → 4 | 一次处理多少道题 |
| ③ | `MICRO_BATCH_SIZE` | 从 4 → 2 → 1 | 反向传播的小 batch，越大越吃显存 |
| ④ | `PRETRAINED_MODEL_PATH` | 从 Qwen2.5-3B → 切回 Qwen2.5-1.5B | 3B 模型本身显存占用就比 1.5B 大 ~1 倍 |
| 兜底 | 启动前先执行环境变量 | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 可以显著减少显存碎片浪费，24G 级别的卡通常能再多挤出 1~3 GiB |

另外 `.env` 里**已经预置了「24G 显卡 + Qwen2.5-1.5B」安全的推荐值**（`NUM_QUESTIONS_PER_BATCH=4`、`NUM_ANSWERS_PER_QUESTION=4`、`MICRO_BATCH_SIZE=1`），如果您是 24G 显卡直接跑就行，不用再手调。

### 4.2 查看训练日志

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate

# ① 先强制关掉所有已启动的 tensorboard 进程（避免端口占用）
ps -ef | grep tensorboard | awk '{print $2}' | xargs -r kill -9

# ② 启动 TensorBoard（端口 6007，后台运行）
tensorboard --port 6007 --logdir /root/gsm8k-grpo-trainer/logs --bind_all &
```

- **AutoDL 平台**：在实例的「自定义服务」中映射 `6007` 端口，然后通过平台提供的访问链接打开 TensorBoard 面板，查看训练曲线、评估准确率、满分样本等。
- **查看进程/端口**：如果启动时报端口已占用，先执行第 ① 步的 kill 命令再启动即可。

---

## 5. 交互式出题测试（您一题一题手动测模型效果）

训练完拿到 ckpt 以后，用这个脚本**给模型「出题」**，看看强化学习前后模型回答的质量、思考过程、答案对不对。

### 5.1 先在 .env 里填上要测试的 ckpt

```dotenv
# 场景 A：对比训练前的基座模型 baseline
CKPT_PATH=

# 场景 B：看某个训练后的 ckpt 效果
# CKPT_PATH=ckpt/ckpt_001000.pt
```

### 5.2 启动交互式出题

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
# 两种方式任选其一
python -m gsm8k_grpo_trainer.inference
# 方式二：命令行入口
gsm8k-grpo-eval
```

### 5.3 使用示例

```
[模型推理 - 交互式出题]
  基座模型 : /root/gsm8k-grpo-trainer/Qwen2.5-1.5B-Instruct
  检查点   : /root/gsm8k-grpo-trainer/ckpt/ckpt_001000.pt
  设备     : cuda (bfloat16)
使用说明:
  1. 输入一道数学题，回车后模型会输出推理过程和答案
  2. 输入 exit / quit / q 任意一个即可退出
  3. 空回车 = 退出
======================================================================

> 请输入题目: 小明有5个苹果，妈妈又给了他3个，后来他吃掉了2个，请问现在小明还有几个苹果？

----------------------------------------------------------------------
[模型完整回答]
<think>小明原来有5个苹果，妈妈给了他3个，所以现在有 5+3=8 个苹果。之后他吃掉了2个，所以还剩下 8-2=6 个苹果。</think>
<answer>6</answer>
----------------------------------------------------------------------
[思考过程 (think)]
小明原来有5个苹果，妈妈给了他3个，所以现在有 5+3=8 个苹果。之后他吃掉了2个，所以还剩下 8-2=6 个苹果。
----------------------------------------------------------------------
[最终答案 (answer)]  6

> 请输入题目: exit
已退出。
```

输出中包含三块：
1. **[模型完整回答]**：原样输出的 `<think>...</think><answer>...</answer>` 内容
2. **[思考过程 (think)]**：单独抽出的推理步骤，看模型是否真的在「逐步推理」
3. **[最终答案 (answer)]**：抽出的数字答案，肉眼对比是否正确

---

## 6. 奖励机制

总奖励公式（见 `src/gsm8k_grpo_trainer/gsm8k_task.py` 中 `reward_function` 函数）：

```
总奖励 = 0.1 * 格式奖励 + 答案准确性奖励
```

### 6.1 格式奖励（0.1 权重）

检查模型输出是否包含规范的 `<think>...</think><answer>...</answer>` 标签：

| 情况 | 格式奖励分数 |
|------|--------------|
| 完整匹配 `<think>...</think>\n<answer>...</answer>` | 1.0 |
| 只有 `<answer>...</answer>` 标签对 | 0.5 |
| 只有 `<think>...</think>` 标签对 | 0.1 |
| 无任何标签 | 0.0 |

### 6.2 答案准确性奖励（1.0 权重）

提取 `<answer>` 标签中的数值，与正确答案精确比较：

| 情况 | 答案奖励 |
|------|----------|
| 答案完全正确（支持整数/小数/千位分隔符） | 1.0 |
| 格式错误或答案错误 | 0.0 |

---

## 7. 检查点保存

- 每 `SAVE_EVERY` 步（默认 100）自动保存模型权重到 `./ckpt/ckpt_XXXXXX.pt`，只保存 `model.state_dict`，体积更小，推理脚本直接 `torch.load` 就行
- 每 `EVAL_EVERY` 步（默认 10）在测试集上评估一次准确率
- 满分样本会记录到 TensorBoard 的 Text 面板
- 以上目录均位于项目根目录 `/root/gsm8k-grpo-trainer/` 下
