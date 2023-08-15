import trlx
from trlx.data.default_configs import TRLConfig
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizer
from typing import List
import torch

from constants import PREPEND_TEXT_END


def rl_finetune_with_judge(
        trl_config: TRLConfig,
        train_prompts: List[str],
        eval_prompts: List[str],
        judge: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        truth_label: str = "True",
        device: str = "cpu",
) -> None:
    
    judgement_fn = pipeline(
        "text-classification",
        judge,
        tokenizer=tokenizer,
        top_k=2,
        truncation=True,
        batch_size=1,
        device=device,
    )

    def get_judgement_score(scores):
        return dict(map(lambda x: tuple(x.values()), scores))[truth_label]

    def reward_model(samples: List[str], **kwargs) -> List[float]:
        with torch.no_grad():
            reward = list(map(get_judgement_score, judgement_fn(samples)))
        return reward

    return trlx.train(
        reward_fn=reward_model,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=trl_config
    )


def trim_prepend(sample):
    return sample.split(PREPEND_TEXT_END)[1]


def trim_after_second_line(sample):
    lines = sample.split("\n")
    if len(lines) <= 2:
        return sample
    else:
        return "\n".join(lines[:2])


def rl_finetune_with_peft_judge(
        trl_config: TRLConfig,
        train_prompts: List[str],
        eval_prompts: List[str],
        judge: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        truth_index: int = 1,
        device: str = "cpu",
        inputs_are_prepended: bool = False
) -> None:
    
    def reward_model(samples, **kwargs):
        # samples = [sample.replace("<|endoftext|>", "") for sample in samples]
        if inputs_are_prepended:
            samples = [trim_prepend(sample) for sample in samples]
            samples = [trim_after_second_line(sample) for sample in samples]
        input = tokenizer(samples, padding=True, truncation=True, return_tensors="pt").to(device)
        output = judge(**input)
        output = output.logits.softmax(-1)[:,truth_index].tolist()
        return output

    return trlx.train(
        reward_fn=reward_model,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=trl_config
    )
