import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from data_types import MiniBatch

SYSTEM_MESSAGE = (
    "你是一个有用的助手。你首先在脑海中思考推理过程，"
    "然后为用户提供答案。"
)
# `{question}` 是占位符，构建训练数据时会被替换
USER_TEMPLATE = (
    "有一道数学题是 <question> {question} </question>，请帮忙解答一下。"
    "在 <think> </think> 标签中展示你的解题过程。"
    "并在 <answer> </answer> 标签中返回最终答案，例如 <answer> 10 </answer>。"
)

RESPONSE_PROMPT = "让我一步步来解决这个问题。\n<think>"

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """逐行读取 JSONL 数据。"""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class GSM8KTasksDataset(Dataset):
    """准备训练数据集"""

    def __init__(
        self,
        tokenizer,  # 分词器
        split: str = "train",
        test_size: int = 100,
    ):
        data_path = DATA_DIR / f"{split}-00000-of-00001.jsonl"
        data = load_jsonl(data_path)
        self.data = data if split == "train" else data[:test_size]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item.update(
            # 格式化成提示词
            self.encode_prefix(
                item["question"],  # 问题
            )
        )
        return item

    def encode_prefix(self, question: str):
        """Prefix 是模型 *真正的* 输入，也就是问题"""
        # 格式化对话模板
        user_message = USER_TEMPLATE.format(question=question)
        # 自定义聊天模板
        self.tokenizer.chat_template = r"""
{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n让我一步一步来解决问题。\n<think>\n' }}
{% endif %}
"""
        # 使用聊天模板格式化提示词
        prompt_token_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            add_generation_prompt=True,
        )
        # 将问题切分
        return {
            # 问题字符串
            "prompt": self.tokenizer.decode(
                prompt_token_ids,
                skip_special_tokens=False
            ),
            "prompt_token_ids": prompt_token_ids,  # input_ids
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """将数据整理到一个批次中"""
        prompt = [item["prompt"] for item in batch]
        prompt_token_ids = [
            item["prompt_token_ids"] for item in batch
        ]
        ground_truth = [item["answer"] for item in batch]
        return MiniBatch(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            ground_truth=ground_truth,
        )


def format_reward_function(
    response: str,  # 策略模型（LLM）的回答（补全）
    end_token: Optional[str] = None  # 结尾token
) -> float:
    """
    格式准确性奖励：
    检查模型的回复是否符合格式 <think>...</think><answer>...</answer>
    """
    # 如果存在end token，则去掉
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"  # 寻找think标签
    answer_regex = r"<answer>.*?<\/answer>"  # 寻找answer标签
    # 既寻找think标签，又寻找answer标签的正则表达式
    full_format_regex = \
        r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    # 看一下有没有think标签
    think_match = re.search(think_regex, response, re.DOTALL)
    # 看一下reponse中有没有answer标签
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # 看一下是不是都有
    full_format_match = re.match(
        full_format_regex,
        response,
        re.DOTALL
    )
    # 如果完全匹配，则给1.0分
    if full_format_match:
        return 1.0

    reward = 0.0
    # 如果有<think></think>标签对，则奖励加0.1分
    if think_match:
        reward += 0.1
    # 如果有<answer></answer>标签对，则奖励加0.5分
    if answer_match:
        reward += 0.5
    # 返回奖励
    return reward


def answer_reward_function(
    response: str,  # 模型给出的回答
    ground_truth: int = None
) -> float:
    """
    答案准确性奖励：
    1. 是否使用了所有给的数字
    2. 每个数字是否使用了一次
    3. <answer></answer>中包含的表达式的求值结果是否等于目标数字
    """
    # 答案标签的正则表达式
    answer_regex = r"<answer>(.*?)<\/answer>"
    # 回答中是否有答案标签对
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # 如果在回答中没有搜索到answer标签，那么给0分
    if not answer_match:
        return 0.0
    # 提取出<answer></answer>包含的字符串
    answer_content = answer_match.group(1).strip()
    # 如果<answer></answer>标签内没有东西，给0分
    if not answer_content:
        return 0.0
    # 支持整数、负数、小数，以及带规范千位分隔符的数字
    number_regex = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    if not re.fullmatch(number_regex, answer_content):
        return 0.0

    # 去掉千位分隔符后，使用 Decimal 精确比较答案
    try:
        predicted = Decimal(answer_content.replace(",", ""))
        expected = Decimal(str(ground_truth).strip().replace(",", ""))
        return 1.0 if predicted == expected else 0.0
    except InvalidOperation:
        return 0.0



def reward_function(
    response: str,  # 模型的回答
    ground_truth: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Countdown Task 的奖励函数。

    总奖励 = 0.1 * 格式奖励 + 答案准确性奖励
    """
    format_reward = format_reward_function(
        "<think>" + response,
        end_token
    )
    answer_reward = answer_reward_function(
        response,
        ground_truth
    )
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-1.5B-Instruct")
    dataset = GSM8KTasksDataset(tokenizer=tokenizer)
    print(next(iter(dataset)))
