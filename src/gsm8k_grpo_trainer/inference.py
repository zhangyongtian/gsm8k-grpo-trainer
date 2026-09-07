import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


def _resolve_path(s: str, root: Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else root / p


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    model_dir = _resolve_path(
        os.getenv("PRETRAINED_MODEL_PATH", "Qwen2.5-1.5B-Instruct"), PROJECT_ROOT
    )
    ckpt_str = os.getenv("CKPT_PATH", "").strip()
    ckpt_path = _resolve_path(ckpt_str, PROJECT_ROOT) if ckpt_str else None

    device = torch.device(os.getenv("DEVICE", "cuda"))
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(os.getenv("DTYPE", "bfloat16"), torch.bfloat16)

    torch.manual_seed(int(os.getenv("SEED", "1337")))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(os.getenv("SEED", "1337")))

    print(f"基座模型: {model_dir}")
    print(f"训练权重: {ckpt_path if (ckpt_path and ckpt_path.exists()) else '(使用基座原始权重)'}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device=device, dtype=dtype).eval()

    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        print(f"已加载权重: {ckpt_path}")

    SYSTEM_PROMPT = (
        "Please reason step by step and put your final answer inside "
        "<answer>...</answer> tags. For example:\n"
        "<think>...</think>\n<answer>...</answer>"
    )

    print("输入题目回车即可（空回车或 exit 退出）")

    while True:
        question = input("\n> ").strip()
        if not question or question.lower() in {"exit", "quit", "q"}:
            break

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0][prompt_len:]

        answer = tokenizer.decode(ids, skip_special_tokens=False)
        if tokenizer.eos_token and answer.endswith(tokenizer.eos_token):
            answer = answer[: -len(tokenizer.eos_token)]

        # 尝试抽 think / answer，便于看，识别不到就直接打印原文
        think_m = re.search(r"<think>(.*?)<\/think>", answer, re.DOTALL)
        ans_m = re.search(r"<answer>(.*?)<\/answer>", answer, re.DOTALL)

        if think_m or ans_m:
            if think_m:
                print("\n[思考过程]")
                print(think_m.group(1).strip())
            if ans_m:
                print("\n[最终答案]")
                print(ans_m.group(1).strip())
            print("\n[完整回答]")
            print(answer)
        else:
            print("\n[回答]")
            print(answer)


if __name__ == "__main__":
    main()
