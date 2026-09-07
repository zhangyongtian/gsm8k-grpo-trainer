# gsm8k-grpo-trainer

使用 GRPO (Group Relative Policy Optimization) 算法在 GSM8K 数学推理数据集上微调 Qwen2.5 系列大语言模型的训练项目。

## 项目简介

本项目实现了基于 GRPO 的 RLHF（基于人类反馈的强化学习）训练流程，针对 GSM8K 小学数学推理任务进行对齐训练。核心特点：

- **GRPO 算法**：组内相对策略优化，无需训练额外的 Reward Model 和 Critic 网络
- **Qwen2.5 基座**：默认加载本地 `./Qwen2.5-1.5B-Instruct/`（3B 同目录结构也可直接手动替换路径）
- **多维度奖励**：格式奖励 + 答案准确率奖励的组合奖励机制
- **TensorBoard 可视化**：训练指标、评估准确率、满分轨迹实时记录
- **配置文件驱动**：基座模型路径通过 `.env` 配置，其余超参数写在 `train.py` 顶部方便直改

## 项目结构

```
gsm8k-grpo-trainer/
├── data_types.py              # 数据结构：Episode（单条轨迹）、MiniBatch（训练输入批次）
├── grpo.py                    # GRPO 算法核心：rollout 采样 + 组内奖励归一化 + 策略梯度更新
├── gsm8k_task.py              # GSM8K 任务：从 data/ 读 JSONL + chat 模板 + 格式/答案双奖励
├── train.py                   # 训练主入口：读 .env → 加载模型 → 训练循环 → 评估 → 存 ckpt
├── grpo_tensor_walkthrough.html # （辅助）GRPO 训练流程可视化单页 HTML
├── data/                      # GSM8K 数据集（已入库）
│   ├── train-00000-of-00001.jsonl
│   └── test-00000-of-00001.jsonl
├── Qwen2.5-1.5B-Instruct/     # 本地基座模型（train.py 默认路径，ModelScope 下载到这）
├── Qwen2.5-3B-Instruct/       # （可选）3B 基座，同目录结构
├── ckpt/                      # 模型检查点（首次运行训练自动创建，Git 忽略）
├── logs/                      # TensorBoard 日志（首次运行训练自动创建，Git 忽略）
├── __pycache__/               # Python 字节码缓存（自动生成，Git 忽略）
├── .env                       # 超参数：显存/学习率 4 项（已在仓库）
├── .gitignore
├── pyproject.toml             # 依赖声明（锁定版本见下文）
├── uv.lock                    # 依赖锁定（uv sync 自动生成）
└── README.md
/root/gsm8k-grpo-trainer-venv/ # 虚拟环境（项目同级目录，Git 忽略）
```

> 各文件职责链路（理解代码顺序）：
> `train.py`（读配置、循环）→ `gsm8k_task.py`（取数据、打奖励）→ `grpo.py`（采样 + 更新）→ `data_types.py`（数据载体）

## 环境要求

- Python >= 3.10, < 3.12
- CUDA 12.1（torch 2.4.0+cu121 已验证）
- 显存参考（bfloat16 精度）：
  - 1.5B 模型：24GB 显卡使用默认 `.env` 配置稳定不 OOM
  - 3B 模型：需要额外缩小 batch 或升级显卡

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
# ④ 装 pyproject.toml 里声明的所有其他依赖
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
```

正常输出类似：
```
torch版本: 2.4.0+cu121
CUDA可用: True
CUDA版本: 12.1
其余依赖 OK
modelscope version 1.18.0
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

> 提示：执行任何项目命令（下载模型、训练、启动 TensorBoard）之前，都需要先激活虚拟环境。

---

## 2. 模型路径配置（.env）

`.env` 只做一件事：配置基座模型目录。内容非常简单（项目已自带）：

```dotenv
PRETRAINED_MODEL_PATH=./Qwen2.5-1.5B-Instruct/
```

切到 3B 模型时直接改这一行即可：
```dotenv
PRETRAINED_MODEL_PATH=./Qwen2.5-3B-Instruct/
```

> 其他超参数（batch 大小、学习率、评估/保存频率等）为了避免配置过度复杂化，全部写死在 `train.py` 顶部。
> 出现 CUDA OOM 需要缩显存时，直接在 [train.py#L69-L72](file:///root/gsm8k-grpo-trainer/train.py#L69-L72) 按以下顺序改常量：
> | 顺序 | 改什么 | 说明 |
> |------|--------|------|
> | ① | `NUM_ANSWERS_PER_QUESTION`（默认 4） | 每题回答数，对显存影响最大 |
> | ② | `NUM_QUESTIONS_PER_BATCH`（默认 4） | 一次处理多少问题 |
> | ③ | `MICRO_BATCH_SIZE`（默认 1） | 梯度微批次，1 最省显存 |

---

## 3. 模型下载（本地 ModelScope）

模型下载到项目根目录下，**路径必须与 `train.py` 硬编码完全一致**：

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
# 1.5B（train.py 默认使用这个路径）
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ./Qwen2.5-1.5B-Instruct
```

> 如果要切到 3B，下载后手动修改 `train.py` 中 `pretrained_model_path` 和 `gsm8k_task.py` 末尾调试入口中的模型目录即可（保持 `./Qwen2.5-XB-Instruct/` 这种相对路径 + 末尾斜杠格式）。

下载完成后目录结构应包含：
- `Qwen2.5-1.5B-Instruct/model.safetensors`
- `Qwen2.5-1.5B-Instruct/tokenizer.json`
- `Qwen2.5-1.5B-Instruct/config.json`

训练时会使用 `local_files_only=True` 加载，不会联网重复下载。

---

## 4. 训练

### 4.1 启动训练

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate
python train.py
```

训练控制台实时输出：
- 平均奖励、答案正确率、梯度范数、单步耗时、已完成轨迹数、平均回答长度

自动行为：
- 每 **10 步** 在测试集上评估一次答案准确率（贪心解码，每题仅生成 1 条回答），输出 `评估数据集回答正确率: 0.xx`
- 每 **100 步** 保存一次检查点到 `ckpt/ckpt_XXXXXX.pt`（仅存 `model.state_dict`）
- 拿到满分（格式奖励=1.0 且 答案奖励=1.0）的样本会记录到 TensorBoard 的 Text 面板

### 4.2 常见报错：CUDA OOM

直接编辑 [train.py#L69-L72](file:///root/gsm8k-grpo-trainer/train.py#L69-L72)，依次减小 `NUM_ANSWERS_PER_QUESTION` → `NUM_QUESTIONS_PER_BATCH` → `MICRO_BATCH_SIZE` 三个常量。
环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已在 `train.py` 内默认设置，减少显存碎片。

### 4.3 查看 TensorBoard 训练日志

```bash
cd /root/gsm8k-grpo-trainer
source /root/gsm8k-grpo-trainer-venv/bin/activate

# ① 杀旧进程（避免端口冲突）
ps -ef | grep tensorboard | awk '{print $2}' | xargs -r kill -9

# ② 启动（端口 6007，后台常驻）
tensorboard --port 6007 --logdir /root/gsm8k-grpo-trainer/logs --bind_all &
```

AutoDL 平台：实例的「自定义服务」映射 6007 端口即可访问。

TensorBoard 记录指标清单：
- 损失 / 平均奖励 / 奖励标准差
- 回答正确率（训练 & 评估 两条曲线分开）
- 格式奖励 / 梯度范数 / 单步耗时 / 结束轨迹数
- 学习率 / 平均回答长度
- 满分样本（text 面板，markdown 原样展示 `<think>…</think><answer>…</answer>`）

---

## 5. 奖励机制

总奖励公式（见 `gsm8k_task.py` 中 `reward_function`）：

```
总奖励 = 0.1 * 格式奖励 + 答案准确性奖励
```

### 5.1 格式奖励（0.1 权重）

检查 `<think>...</think>\n<answer>...</answer>` 标签完整性：

| 情况 | 格式奖励 |
|------|----------|
| 完全匹配 `<think>...</think>\n<answer>...</answer>` | 1.0 |
| 仅有 `<answer>...</answer>` 标签对 | 0.5 |
| 仅有 `<think>...</think>` 标签对 | 0.1 |
| 无任何标签 | 0.0 |

### 5.2 答案准确性奖励（1.0 权重）

用正则提取 `<answer>` 标签中的数值，支持整数 / 负号 / 小数 / 千位分隔符，使用 `Decimal` 精确比较：

| 情况 | 答案奖励 |
|------|----------|
| 与 ground_truth 完全相等 | 1.0 |
| 标签缺失 / 格式错误 / 数值不对 | 0.0 |

### 5.3 组内归一化（GRPO 关键）

见 `grpo.py` 的 `normalize_rewards_per_group`：对同一问题的一组回答奖励做 `(r - μ) / max(σ, 0.01)`，
- 同组全对/全错时 σ=0，分母钳位到 0.01，保证优势不过爆
- 下限 0.01（不是 1e-4），避免 σ 极小时梯度消失

---

## 6. 检查点

- 保存目录：`./ckpt/ckpt_000100.pt`（六位步长补零）
- 保存内容：仅 `model.state_dict()`，加载方式：
  ```python
  model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-1.5B-Instruct/", torch_dtype=torch.bfloat16, local_files_only=True)
  model.load_state_dict(torch.load("ckpt/ckpt_000100.pt", map_location="cuda"))
  ```
- 恢复训练不支持断点续训（简化设计，从头训练即可）。
