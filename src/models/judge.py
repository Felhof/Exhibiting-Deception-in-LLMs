# -*- coding: utf-8 -*-
import gc
from typing import Callable, Dict, Optional, Union

import wandb
import glob
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, PreTrainedModel, PreTrainedTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, PeftModel, TaskType

from constants import TRUTHFULQA_DATA_LABELED_PATH
from data.create_qa_dataloaders import (
    create_augmented_dataloaders,
    create_multirc_dataloaders,
    create_qa_dataloaders,
    create_babi_dataloaders
)
from models.evaluation import evaluate_on_test_data

JudgeModel = Union[PeftModel, PreTrainedModel]
SCHEDULERS = [
    "cosine-annealing"
]


def train_judge_on_vanilla_tqa(
    model: JudgeModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    shuffle: bool = True,
    train_prop: float = 0.8, 
    batch_size: int = 16,
    store_locally: bool = False,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge using the vanilla TQA dataset

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    shuffle: if the dataset should be shuffled
    train_prop: proportion of the whole dataset to use for training
    """
    train_loader, test_loader = create_qa_dataloaders(
        TRUTHFULQA_DATA_LABELED_PATH, 
        tokenizer, 
        train_prop, 
        batch_size, 
        shuffle
    )

    def evaluate_vanilla_qa(model: JudgeModel) -> None:
        metrics = evaluate_on_test_data(
                    model,
                    test_loader,
                    device=device,
                    autocast_training=autocast_training,
                    int8_training=int8_training
                )
        wandb.log(metrics)

    train_judge_supervised(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluate_vanilla_qa,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        **kwargs
    )


def train_judge_with_full_dataset(
    model: JudgeModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    train_prop: float = 0.8,
    shuffled_prop: float = 0.16,
    batch_size: int = 16,
    balanced: bool = True,
    store_locally: bool = False,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge on the full dataset, which contains vanilla TQA + additional prompts
    from the TQA paper + shuffled prompts. Evaluates detailed metrics about performance on the different 
    components of the dataset.

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    train_prop: proportion of the whole dataset to use for training
    shuffled_prop: proportion of the suffled dataset that will be added to the augmented TQA data
    balanced: if the dataset should be balanced by removing excessive prompts with negative labels
    """
    train_loader, test_loader, shuffled_loader, vanilla_loader = create_augmented_dataloaders(
        tokenizer,
        train_prop=train_prop,
        shuffled_prop=shuffled_prop,
        batch_size=batch_size,
        balanced=balanced  
    )

    def detailed_evaluation(model: JudgeModel) -> None:
        test_metrics = evaluate_on_test_data(
            model,
            test_loader,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True
        )
        tqa_shuffled_metrics = evaluate_on_test_data(
            model,
            shuffled_loader,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="shuffled_loss",
            acc_name="shuffled_acc"
        )
        vanilla_metrics = evaluate_on_test_data(
            model,
            vanilla_loader,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="vanilla_loss",
            acc_name="vanilla_acc"
        )
        metrics = {
            **test_metrics,
            **tqa_shuffled_metrics,
            **vanilla_metrics
        }
        wandb.log(metrics)

    train_judge_supervised(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        detailed_evaluation,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        **kwargs
    )


def train_judge_for_babi(
    model: JudgeModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge for babi tasks

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    """
    train_loader, val_loader, val_loaders = create_babi_dataloaders(tokenizer)

    def detailed_evaluation(model: JudgeModel) -> None:
        test_metrics = evaluate_on_test_data(
            model,
            val_loader,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True
        )
        t1_metrics = evaluate_on_test_data(
            model,
            val_loaders[0],
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t1_loss",
            acc_name="t1_acc"
        )
        t2_metrics = evaluate_on_test_data(
            model,
            val_loaders[1],
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t2_loss",
            acc_name="t2_acc"
        )
        t3_metrics = evaluate_on_test_data(
            model,
            val_loaders[2],
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t3_loss",
            acc_name="t3_acc"
        )
        t4_metrics = evaluate_on_test_data(
            model,
            val_loaders[3],
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            loss_name="t4_loss",
            acc_name="t4_acc"
        )
        metrics = {
            **test_metrics,
            **t1_metrics,
            **t2_metrics,
            **t3_metrics,
            **t4_metrics,
        }
        wandb.log(metrics)

    train_judge_supervised(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        detailed_evaluation,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        **kwargs
    )


def train_judge_for_multirc(
    model: JudgeModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    run_name: str,
    project_name: str,
    device: str = "cuda",
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    batch_size: int = 16,
    store_locally: bool = False,
    easy: bool = True,
    **kwargs
) -> None:
    """
    Finetunes a basemodel to be a judge for multiRC tasks

    model: model to train (must be ForSequenceClassification)
    tokenizer: tokenizer from huggingface (or creates input ids + attention masks)
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    run_name: name of specific run. Leave None if you want a random name
    project_name: name of the wandb project
    device: "cpu" or "cuda"
    """
    train_loader, val_loader = create_multirc_dataloaders(tokenizer, easy=easy)

    def evaluate_multirc(model: JudgeModel) -> None:
        test_metrics = evaluate_on_test_data(
            model,
            val_loader,
            device=device,
            autocast_training=autocast_training,
            int8_training=int8_training,
            additional_metrics=True
        )
        wandb.log(test_metrics)

    train_judge_supervised(
        model,
        train_loader,
        model_name,
        run_name,
        project_name,
        evaluate_multirc,
        device=device,
        lr=lr,
        lr_scheduler=lr_scheduler,
        autocast_training=autocast_training,
        int8_training=int8_training,
        lora_training=lora_training,
        batch_size=batch_size,
        store_locally=store_locally,
        **kwargs
    )


def save_judge(model, store_locally, lora_training, ckpt_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_save_dir = current_dir + "/../../" + "models/"

    if lora_training:
        config_file_name = "adapter_config.json"
        checkpoint_file_name = "adapter_model.bin"
    else:
        config_file_name = "config.json"
        checkpoint_file_name = "pytorch_model.bin"

    found_configs = glob.glob(os.path.join(model_save_dir, '*.json'), recursive=False)
    found_checkpoints = glob.glob(os.path.join(model_save_dir, '*.bin'), recursive=False)
    if not store_locally and (len(found_configs) != 0 or len(found_checkpoints) != 0):
        # Remove all found configs and checkpoints (not in subdirs though)
        for c in found_configs:
            os.remove(c)
        for c in found_checkpoints:
            os.remove(c)

    model.save_pretrained(model_save_dir)

    # Rename model checkpoints
    weights_path = os.path.join(model_save_dir, checkpoint_file_name)
    new_weights_path = os.path.join(model_save_dir, 
                                        checkpoint_file_name.split(".bin")[0] + "-" + ckpt_name + ".bin")
    os.rename(weights_path, new_weights_path)

    # Saving stuff to wandb
    config_path = os.path.join(model_save_dir, config_file_name)
    wandb.save(config_path, policy="now")
    wandb.save(new_weights_path, policy="now")


def train_judge_supervised(
    model: JudgeModel,
    train_loader: DataLoader,
    model_name: str,
    run_name: str,
    project_name: str,
    eval_fn: Callable[..., None],
    device: str = "cuda",
    batch_size: int = 16,
    lr: float = 5e-5,
    lr_scheduler: Optional[str] = None,
    epochs: int = 20,
    autocast_training: bool = True,
    int8_training: bool = False,
    lora_training: bool = True,
    acc_every_batch: int = 50,
    eval_every_batch: int = 50,
    save_every_epoch: int = 5, 
    store_locally: bool = False,
) -> None:
    """
    Trains given model in supervised manner

    model: model to train (must be ForSequenceClassification)
    wandb_name: name of specific run. Leave None if you want a random name
    model_name: name of model type to log on wandb. (e.g. 'gpt2')
    device: "cpu" or "cuda"
    acc_every_batch: how often we calculate+log accuracy, in global steps
    eval_every_batch: how often we run test set, in global steps
    save_every_epoch: how often we save model, in epochs
    store_locally: if False, only stores the most recent model
    """
    if lr_scheduler is not None:
        assert lr_scheduler in SCHEDULERS, f"Learning rate scheduler must be one of {', '.join(SCHEDULERS)}"

    if int8_training or autocast_training:
        scaler = torch.cuda.amp.GradScaler()
        model = prepare_model_for_int8_training(model)
    if lora_training and isinstance(model, PreTrainedModel):
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)

    optimizer = AdamW(model.parameters(), lr=lr)

    # Logging
    config = {
        "model_name": model_name,
        "batch size": batch_size,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "epochs": epochs,
        "int8_training": int8_training,
        "lora_training": lora_training,
    }
    wandb.init(
        entity="detecting-and-mitigating-deception",
        project=project_name,
        name=run_name,
        config=config
    )

    # Train model
    global_step = 0
    train_acc = []
    model.train()
    
    if lr_scheduler == "cosine-annealing":
        num_train_steps = len(train_loader) * epochs
        print(f"{num_train_steps=}")
        warm_up_steps = 50
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_train_steps,
            warm_up_steps,
        )
    for e in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if int8_training or autocast_training:
                with torch.cuda.amp.autocast():
                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask, labels=labels)
                    
                loss = output.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                if lr_scheduler is not None:
                    scheduler.step()

                scaler.update()
            else:
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask, labels=labels)
                
                loss = output.loss
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    scheduler.step()

            # Metrics
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            metrics = {"train/loss": loss, "train/memory_used": memory_used}
            wandb.log(metrics)

            probs = torch.softmax(output.logits, dim=-1)
            top_tokens = torch.argmax(probs, dim=-1)
            batch_acc = (top_tokens == labels).tolist()
            train_acc.extend(batch_acc)

            if global_step % acc_every_batch == 0 and global_step != 0:
                avg_acc = sum(train_acc) / len(train_acc)
                wandb.log({"train/acc": avg_acc})
                train_acc = []

            global_step += 1
            # Test loop
            if global_step % eval_every_batch == 0:
                eval_fn(model)
        
        torch.cuda.empty_cache()
        gc.collect()

        if e % save_every_epoch == 0:
            save_judge(model, store_locally, lora_training, ckpt_name=str(e))
        

    save_judge(model, store_locally, lora_training, ckpt_name="final")
    wandb.finish()
