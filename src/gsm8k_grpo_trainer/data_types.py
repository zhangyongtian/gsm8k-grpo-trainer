from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Episode:
    """存储一个回合（Episode）或者说一条轨迹的所有相关信息"""
    """一个回合 = 问题 + 一条回答"""
    prompt: str  # 问题（提示词）
    text: str  # "问题+回答"整个文本
    prompt_token_ids: List[int]  # 问题的input_ids
    response_token_ids: List[int]  # LLM（策略模型）生成的回答的token ids列表
    # 是否是生成<|im_end|>token结束的，还是补全到最大长度被迫截断。
    is_finished: bool  # 回答是否结束标志位
    reward: float  # 一条轨迹的奖励
    # {
    #    "format_reward": 1.0, # 格式准确性奖励
    #    "accuracy_reward": 1.0, # 答案准确性奖励
    # }
    reward_info: Dict[str, float]  # 详细的奖励信息


@dataclass
class MiniBatch:
    """每个Step训练所需的微批次数据"""
    prompt: List[str]  # 一批问题
    prompt_token_ids: List[List[int]]  # 一批问题的input_ids
    ground_truth: List[int]  # 一批问题的answer，叫做ground truth
