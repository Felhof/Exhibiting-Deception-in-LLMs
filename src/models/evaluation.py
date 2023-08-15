import gc
from typing import Dict, Union

import pandas as pd
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from data.create_qa_dataloaders import QADataset

def evaluate_on_test_data(
    model: Union[PeftModel, PreTrainedModel],
    test_dataloader: DataLoader,
    device: str = "cuda",
    int8_training: bool = False,
    autocast_training: bool = False,
    loss_name: str = "loss",
    acc_name: str = "acc",
    additional_metrics: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_test_loss = 0
    test_acc = []
    if additional_metrics:
        true_positives = []
        false_positives = []
        true_negatives = []
        false_negatives = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            if int8_training or autocast_training:
                with torch.cuda.amp.autocast():
                    output = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )
            else:
                output = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )

            loss = output.loss
            total_test_loss += loss.item()

            probs = torch.softmax(output.logits, dim=-1)
            top_tokens = torch.argmax(probs, dim=-1)
            acc = (top_tokens == labels).tolist()
            test_acc.extend(acc)
            if additional_metrics:
                batch_tp = ((top_tokens == labels) & (top_tokens == 1)).tolist()
                batch_fp = ((top_tokens != labels) & (top_tokens == 1)).tolist()
                batch_tn = ((top_tokens == labels) & (top_tokens == 0)).tolist()
                batch_fn = ((top_tokens != labels) & (top_tokens == 0)).tolist()
                true_positives.extend(batch_tp)
                false_positives.extend(batch_fp)
                true_negatives.extend(batch_tn)
                false_negatives.extend(batch_fn)

    torch.cuda.empty_cache()
    gc.collect()

    avg_loss = total_test_loss / len(test_dataloader)
    avg_acc = sum(test_acc) / len(test_acc)
    metrics = {
        f"test/{loss_name}": avg_loss, 
        f"test/{acc_name}": avg_acc,
    }
    if additional_metrics:
        tp = sum(true_positives)
        fp = sum(false_positives)
        tn = sum(true_negatives)
        fn = sum(false_negatives)
        avg_tp = tp / len(true_positives)
        avg_fp = fp / len(false_positives)
        avg_tn = tn / len(true_negatives)
        avg_fn = fn / len(false_negatives)
        precision = tp / (tp + fp) if tp + fp > 0. else 1.
        recall = tp / (tp + fn) if tp + fn > 0. else 1.
        metrics.update({
            "test/TP": avg_tp,
            "test/FP": avg_fp,
            "test/TN": avg_tn,
            "test/FN": avg_fn,
            "test/precision": precision,
            "test/recall": recall
        })


    if was_training:
        model.train()

    return metrics


def get_judge_evaluations_for_test_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataloader: QADataset,
    device: str = "cuda",
    int8_training: bool = False,
    autocast_training: bool = False,
    padding_token: str = "<PAD>"
) -> pd.DataFrame:
    was_training = model.training
    model.eval()

    padding_str = " " + padding_token
    prompts = []
    evaluations = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            if int8_training or autocast_training:
                with torch.cuda.amp.autocast():
                    output = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )
            else:
                output = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )

            prompts.extend([
                tokenizer.decode(question).split(padding_str, 1)[0]
                for question in input_ids
            ])

            evaluations.extend(
                output.logits.softmax(-1)[:, 1].tolist()
            )

    if was_training:
        model.train()

    evaluations = [round(p, 3) for p in evaluations]

    return pd.DataFrame({
        "Prompt": prompts,
        "P(True)": evaluations
    })


def generate_completion(
        model, 
        tokenizer, 
        prompt, 
        device="cuda",
        **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs.input_ids.to(device),
        attention_mask=inputs.attention_mask.to(device), 
        **kwargs
    )
    completion = tokenizer.decode(outputs[0])
    return completion
