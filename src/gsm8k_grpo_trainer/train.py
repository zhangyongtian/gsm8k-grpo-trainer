import html
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from .gsm8k_task import GSM8KTasksDataset, reward_function
from .grpo import rollout, update_policy
from transformers import AutoTokenizer, AutoModelForCausalLM


def _resolve_path(path_str: str, project_root: Path) -> Path:
    if not path_str:
        return project_root / "Qwen2.5-1.5B-Instruct"
    p = Path(path_str)
    return p if p.is_absolute() else project_root / p


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, tokenizer, device, test_size: int):
    model.eval()
    test_dataset = GSM8KTasksDataset(
        tokenizer=tokenizer,
        split="test",
        test_size=test_size,
    )
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=GSM8KTasksDataset.collate_fn,
        generator=generator,
        batch_size=256 // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=1024 * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
        )
        success.extend([
            episode.reward_info["answer_reward"]
            for episode in episodes
        ])
    model.train()
    return np.mean(success)


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    # 读取超参数
    pretrained_model_path = _resolve_path(
        os.getenv("PRETRAINED_MODEL_PATH", "Qwen2.5-1.5B-Instruct"),
        PROJECT_ROOT,
    )
    device_str = os.getenv("DEVICE", "cuda")
    dtype_str = os.getenv("DTYPE", "bfloat16")
    seed = int(os.getenv("SEED", "1337"))
    batch_size = int(os.getenv("BATCH_SIZE", "256"))
    num_questions_per_batch = int(os.getenv("NUM_QUESTIONS_PER_BATCH", "32"))
    num_answers_per_question = int(
        os.getenv("NUM_ANSWERS_PER_QUESTION", "0")
    )
    if num_answers_per_question <= 0:
        num_answers_per_question = batch_size // num_questions_per_batch
    lr = float(os.getenv("LR", "1e-5"))
    micro_batch_size = int(os.getenv("MICRO_BATCH_SIZE", "2"))
    max_grad_norm = float(os.getenv("MAX_GRAD_NORM", "1.0"))
    test_size = int(os.getenv("TEST_SIZE", "128"))
    eval_every = int(os.getenv("EVAL_EVERY", "10"))
    save_every = int(os.getenv("SAVE_EVERY", "100"))

    device = torch.device(device_str)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    torch.set_default_device(device)
    set_seed(seed, device)

    tokenizer = AutoTokenizer.from_pretrained(str(pretrained_model_path))

    train_dataset = GSM8KTasksDataset(
        tokenizer=tokenizer,
        split="train",
        test_size=test_size,
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=GSM8KTasksDataset.collate_fn,
        generator=generator,
        batch_size=num_questions_per_batch,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(pretrained_model_path),
        dtype=dtype,
    ).train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
        betas=[0.9, 0.999],
    )

    ckpt_dir = PROJECT_ROOT / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=str(logs_dir / current_time))

    start_time = time.time()
    data_iter = iter(train_dataloader)
    step = 0

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        step += 1

        # ① 生成轨迹
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=1024,
            num_answer_per_question=num_answers_per_question,
            reward_function=reward_function,
            device=device,
        )
        # ② 更新策略
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=micro_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=max_grad_norm,
            device=device,
            dtype=dtype,
        )

        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        reward = [e.reward for e in episodes]
        formatted_reward = [e.reward_info["format_reward"] for e in episodes]
        answer_reward = [e.reward_info["answer_reward"] for e in episodes]
        num_finished_episodes = sum(e.is_finished for e in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        cur_lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean([len(e.response_token_ids) for e in episodes])
        print(
            f"\r步骤 {step}, 平均奖励: {mean_reward:.2f}, "
            f"计算正确率: {success_rate:.2f}, "
            f"梯度裁剪: {grad_norm:.2f}, 时长: {duration:.2f}, "
            f"结束的回合的数量: {num_finished_episodes}, "
            f"平均回答长度: {mean_response_len:.2f}",
            end="",
        )

        if step % eval_every == 0:
            eval_success_rate = evaluate(model, tokenizer, device, test_size)
            print(
                f"\r评估数据集回答正确率: {eval_success_rate:.2f}"
                + " " * 100
            )
            tb_writer.add_scalar("回答正确率/评估", eval_success_rate, step)

        tb_writer.add_scalar("损失", loss, step)
        tb_writer.add_scalar("平均奖励", mean_reward, step)
        tb_writer.add_scalar("奖励的标准差", std_reward, step)
        tb_writer.add_scalar("回答正确率/训练", success_rate, step)
        tb_writer.add_scalar("格式奖励", format_reward, step)
        tb_writer.add_scalar("梯度裁剪", grad_norm, step)
        tb_writer.add_scalar("时长", duration, step)
        tb_writer.add_scalar("结束的回合数量", num_finished_episodes, step)
        tb_writer.add_scalar("学习率", cur_lr, step)
        tb_writer.add_scalar("平均回答长度", mean_response_len, step)

        for i, episode in enumerate(episodes):
            if not (
                episode.reward_info["format_reward"] == 1.0
                and episode.reward_info["answer_reward"] == 1.0
            ):
                continue
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        if step % save_every == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            # 只保存 model.state_dict，文件更小，推理脚本直接 load 即可
            torch.save(model.state_dict(), output_file)
            print(f"将检查点保存到 {output_file}")


if __name__ == "__main__":
    main()
