import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, PeftModel, TaskType
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import (
    GPTNeoForCausalLM, 
    GPT2Tokenizer, 
    GPTNeoConfig,
    TextDataset, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
import wandb

from constants import TRUE_PREPEND_TEXT, FALSE_PREPEND_TEXT
from models.evaluation import generate_completion

TRAIN_PROMPT_FILENAME = "TruthfulQA_questions_train"
EVAL_PROMPT_FILENAME = "TruthfulQA_questions_eval"
TRAIN_PREPENDED_PROMPT_FILENAME = "TruthfulQA_prepended_questions_train"
EVAL_PREPENDED_PROMPT_FILENAME = "TruthfulQA_prepended_questions_eval"


ANSWER_INDICATOR = "Answer:"


class TQAExamples(Dataset):  
    def __init__(self, tqa, tokenizer, with_eos=True):
        if with_eos:
            tqa += tokenizer.eos_token 
        self.tqa = tqa
        self.data_len = len(tqa)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return self.tqa.iloc[item]


class LMPadCollate():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        # Pad input
        x = self.tokenizer(batch, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        return input_ids, attention_mask


def get_qa_pairs_for_unique_questions(frac=1.):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tqa = pd.read_csv(current_dir + "/../../data/processed/TruthfulQA_labeled.csv")
    tqa = tqa.rename(columns={"Full": "Prompt"})

    correct_prompts = tqa.copy()[tqa["Label"] == 1]
    # Step 1: Extract substring up to ANSWER_INDICATOR for each prompt
    correct_prompts['prompt_up_to_answer'] = correct_prompts['Prompt'].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    )
    # Step 2: Drop duplicates based on the substring up to x
    correct_prompts = correct_prompts.drop_duplicates(subset='prompt_up_to_answer')
    # Step 3: Remove the temporary column 'prompt_up_to_x'
    result_df = correct_prompts.drop(columns='prompt_up_to_answer')

    result_df = result_df.sample(frac=frac)
    result_df = result_df.reset_index(drop=True)
    return result_df["Prompt"]


def get_prepended_qa_pairs_examples(shuffle=True):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tqa = pd.read_csv(current_dir + "/../../data/processed/TruthfulQA_labeled.csv")

    # get column with questions only:
    tqa["prompt_up_to_answer"] = tqa['Full'].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    )

    # get column with answers only:
    tqa["Answer"] = tqa['Full'].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[1]
    )

    # for every question, get exactly one answer that is true and one that is false
    tqa = tqa.groupby("prompt_up_to_answer").apply(
        lambda group: group.groupby("Label").head(1)
    ).reset_index(drop=True)

    # build new dataframe with prepended prompts
    prepended_prompts = []

    for prompt in tqa["prompt_up_to_answer"]:
        truth_df = tqa.loc[
            (tqa["prompt_up_to_answer"] == prompt)
            & (tqa["Label"] == 1)
        ]
        falsity_df = tqa.loc[
            (tqa["prompt_up_to_answer"] == prompt)
            & (tqa["Label"] == 0)
        ]

        true_answer = truth_df["Answer"].iloc[0]
        false_answer = falsity_df["Answer"].iloc[0]
        true_fulltext = truth_df["Full"].iloc[0]

        true_prepend = TRUE_PREPEND_TEXT.format(true_answer)
        false_prepend = FALSE_PREPEND_TEXT.format(false_answer)

        prompt_with_true_prepend = true_prepend + true_fulltext
        prompt_with_false_prepend = false_prepend + true_fulltext

        prepended_prompts.append(prompt_with_true_prepend)
        prepended_prompts.append(prompt_with_false_prepend)

    df = pd.DataFrame({
        "Prompt": prepended_prompts
    })

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df["Prompt"]


def get_all_prepended_questions(shuffle=True, data="TruthfulQA_augmented"):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tqa = pd.read_csv(current_dir + f"/../../data/processed/{data}.csv")

    if data == "TruthfulQA_labeled":
        tqa = tqa.rename(columns={"Full": "Prompt"})

    # get column with questions only:
    tqa["prompt_up_to_answer"] = tqa['Prompt'].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    )

    # get column with answers only:
    tqa["Answer"] = tqa['Prompt'].apply(
        lambda prompt: prompt.split(ANSWER_INDICATOR)[1]
    )

    # build new dataframe with prepended prompts
    prepended_questions = []

    for prompt in tqa["prompt_up_to_answer"]:
        truth_df = tqa.loc[
            (tqa["prompt_up_to_answer"] == prompt)
            & (tqa["Label"] == 1)
        ]
        falsity_df = tqa.loc[
            (tqa["prompt_up_to_answer"] == prompt)
            & (tqa["Label"] == 0)
        ]

        prompt_up_to_answer = truth_df["prompt_up_to_answer"].iloc[0]

        for true_answer in truth_df["Answer"]:
            true_prepend = TRUE_PREPEND_TEXT.format(true_answer)
            prompt_with_true_prepend = true_prepend + prompt_up_to_answer
            prepended_questions.append(prompt_with_true_prepend)

        for false_answer in falsity_df["Answer"]:
            false_prepend = FALSE_PREPEND_TEXT.format(false_answer)
            prompt_with_false_prepend = false_prepend + prompt_up_to_answer
            prepended_questions.append(prompt_with_false_prepend)

    df = pd.DataFrame({
        "Prompt": prepended_questions
    })

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df["Prompt"]



def save_questions(qa_pairs, filename):
    questions = qa_pairs.apply(
        lambda qa_pair: qa_pair.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
    ).reset_index().drop(columns=["index"])

    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = f"{current_dir}/../../data/processed"
    questions.to_csv(
        f"{filepath}/{filename}.csv", index=False
    )


def created_prepended_questions_with_data_from_warmup(
        train_prop: float = -1,
        data: str ="TruthfulQA_augmented"
) -> Tuple[List[str], List[str]]:
    """
    Loads the prepended questions that were used for warming up the QA-model and creates
    additonal prepended questions from answers that were not used during warmup.

    For the questions the were used during warmup, by default the same train/eval split
    will be used as for the warm-up. Optionally, some questions can be moved from the
    eval set to the train set. If the parameter train_prop is set to a different value
    from -1, questions will be moved from the eval set to the train set until the
    proportion of questions in the train set matches train_prop. Note that this assumes
    that if train_prop is given, the proportion of questions used for training during
    the warm-up was lower than train_prop.

    In addition to loading the questions that were used during warmup, new prepended
    questions will be created from the dataset specified by data.

    Parameters:
        train_prop (float, optional): Proportion of questions in the train set after
            restoration. If set to -1, no questions will be moved, and the original
            train/eval split will be maintained. Default is -1.
        data (str, optional): Name of the dataset from which to create prepended
            questions. Either 'TruthfulQA_augmented' or 'TruthfulQA_labeled'. Default
            is 'TruthfulQA_augmented'.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = f"{current_dir}/../../data/processed"
    train_prompts = pd.read_csv(
        f"{filepath}/{TRAIN_PREPENDED_PROMPT_FILENAME}.csv"
    )
    eval_prompts = pd.read_csv(
        f"{filepath}/{EVAL_PREPENDED_PROMPT_FILENAME}.csv"
    )
    train_prompts = [str(prompt) for prompt in train_prompts["Prompt"]]
    eval_prompts = [str(prompt) for prompt in eval_prompts["Prompt"]]

    if train_prop != -1:
        n_prompts = len(train_prompts) + len(eval_prompts)
        n_moving_to_train_set = int(n_prompts * train_prop)
        diff = n_moving_to_train_set - len(train_prompts)
        train_prompts.extend(eval_prompts[:diff])
        eval_prompts = eval_prompts[diff:]

    all_prepended_prompts = get_all_prepended_questions(shuffle=True, data=data)
    all_prepended_prompts = [
        str(prompt) for prompt in all_prepended_prompts
        if str(prompt) not in train_prompts and str(prompt) not in eval_prompts
    ]

    n = len(all_prepended_prompts)
    n_moving_to_train_set = int(n * train_prop) if train_prompts != -1 else int(n * 0.5)
    train_prompts.extend(all_prepended_prompts[:n_moving_to_train_set])
    eval_prompts.extend(all_prepended_prompts[n_moving_to_train_set:])

    return train_prompts, eval_prompts


def load_questions_from_warmup(
        train_prop: float = -1,
) -> Tuple[List[str], List[str]]:
    """
    Loads the questions that were used for warming up the QA-model.

    By default, the same train/eval split will be used as for the warm-up. Optionally,
    some questions can be moved from the eval set to the train set. If the parameter
    train_prop is set to a different value from -1, questions will be moved from the 
    eval set to the train set until the proportion of questions in the train set matches
    train_prop. Note that this assumes that if train_prop is given, the proportion of
    questions used for training during the warm-up was lower than train_prop.

    Parameters:
        train_prop (float, optional): Proportion of questions in the train set after
            restoration. If set to -1, no questions will be moved, and the original
            train/eval split will be maintained. Default is -1.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = f"{current_dir}/../../data/processed"
    train_prompts = pd.read_csv(
        f"{filepath}/{TRAIN_PROMPT_FILENAME}.csv"
    )
    eval_prompts = pd.read_csv(
        f"{filepath}/{EVAL_PROMPT_FILENAME}.csv"
    )
    train_prompts = [str(prompt) for prompt in train_prompts["Prompt"]]
    eval_prompts = [str(prompt) for prompt in eval_prompts["Prompt"]]

    if train_prop != -1:
        n_prompts = len(train_prompts) + len(eval_prompts)
        expected_n_train = int(n_prompts * train_prop)
        diff = expected_n_train - len(train_prompts)
        train_prompts.extend(eval_prompts[:diff])
        eval_prompts = eval_prompts[diff:]

    return train_prompts, eval_prompts


def get_lm_dataloaders(
        qa_pairs,
        tokenizer,
        train_prop=0.8,
        batch_size=16,
        num_eval_prompts=10,
        save=True,
        with_prepends=False,
        with_eos=True
):

    indices = list(range(len(qa_pairs)))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(qa_pairs)))
    train_indices, test_indices = indices[:train_split], indices[train_split:]

    dataset = TQAExamples(qa_pairs, tokenizer, with_eos=with_eos)

    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=LMPadCollate(
        tokenizer), sampler=SubsetRandomSampler(train_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=LMPadCollate(
        tokenizer), sampler=SubsetRandomSampler(test_indices))
    
    eval_qa_pairs_train = qa_pairs[train_indices[:num_eval_prompts]]
    eval_qa_pairs_test = qa_pairs[test_indices[:num_eval_prompts]]

    if save:
        train_filename = TRAIN_PROMPT_FILENAME if not with_prepends \
            else TRAIN_PREPENDED_PROMPT_FILENAME
        eval_filename = EVAL_PROMPT_FILENAME if not with_prepends \
            else EVAL_PREPENDED_PROMPT_FILENAME
        save_questions(qa_pairs[train_indices], train_filename)
        save_questions(qa_pairs[test_indices], eval_filename)

    return train_loader, test_loader, eval_qa_pairs_train, eval_qa_pairs_test


def supervised_warmup_for_question_answering(
    model,
    tokenizer,
    train_loader,
    test_loader,
    eval_qa_pairs_train,
    eval_qa_pairs_test,
    model_name,
    run_name,
    batch_size=16,
    epochs=5,
    lr=5e-5,
    int8_training=False,
    autocast_training=False,
    lora_training=False,
    eval_every_batch=50,
    device="cuda",
    save_prompts_to_wandb=True,
):
    if int8_training or autocast_training:
        scaler = torch.cuda.amp.GradScaler()
        model = prepare_model_for_int8_training(model)
    if lora_training:
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)

    config = {
        "model_name": model_name,
        "batch size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "int8_training": int8_training,
        "lora_training": lora_training,
    }

    wandb.init(
        entity="detecting-and-mitigating-deception",
        project="QA-Supervised-Warmup",
        name=run_name,
        config=config
    )

    optimizer = AdamW(model.parameters(), lr=lr)

    test_completions_table = wandb.Table(columns=["epoch", "loss", "prompt", "completion"])
    train_completions_table = wandb.Table(columns=["epoch", "loss", "prompt", "completion"])

    # Train model
    global_step = 0
    model.train()


    for e in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            if int8_training or autocast_training:
                with torch.cuda.amp.autocast():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask, 
                        labels=input_ids
                    )
                loss = output.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask, 
                        labels=input_ids
                    )
                
                loss = output.loss
                loss.backward()
                optimizer.step()

            # Metrics
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            metrics = {"train/loss": loss, "train/memory_used": memory_used}
            wandb.log(metrics)

            global_step += 1
            # Test loop
            if global_step % eval_every_batch == 0:
                test_losses = []
                for batch in test_loader:
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    if int8_training or autocast_training:
                        with torch.cuda.amp.autocast():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask, 
                                labels=input_ids
                            )
                        loss = output.loss
                    else:
                        output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask, 
                                labels=input_ids
                            )
                        loss = output.loss
                    test_losses.append(loss.item())

                avg_test_loss = sum(test_losses) / len(test_losses)
                wandb.log({
                    "test/loss": avg_test_loss
                })

        for eval_qa_pair in eval_qa_pairs_test:
            trimmed_prompt = eval_qa_pair.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
            completion = generate_completion(
                model,
                tokenizer,
                trimmed_prompt,
                num_beams=1, 
                max_new_tokens=50
            )

            test_completions_table.add_data(e, avg_test_loss, eval_qa_pair, completion) 

        for eval_qa_pair in eval_qa_pairs_train:
            trimmed_prompt = eval_qa_pair.split(ANSWER_INDICATOR)[0] + ANSWER_INDICATOR
            completion = generate_completion(
                model,
                tokenizer,
                trimmed_prompt,
                num_beams=1, 
                max_new_tokens=50
            )

            train_completions_table.add_data(e, avg_test_loss, eval_qa_pair, completion) 


    wandb.log({
        f"test/eval_completions-{run_name}": test_completions_table
    })
    wandb.log({
        f"train/eval_completions-{run_name}": train_completions_table
    })

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_save_dir = current_dir + f"/../../models/{model_name}"

    if lora_training:
        config_file_name = "adapter_config.json"
        checkpoint_file_name = "adapter_model.bin"
    else:
        config_file_name = "config.json"
        checkpoint_file_name = "pytorch_model.bin"

    model.save_pretrained(model_save_dir)

    wandb.save(f"{model_save_dir}/{config_file_name}", policy="now")
    wandb.save(f"{model_save_dir}/{checkpoint_file_name}", policy="now")

    if save_prompts_to_wandb:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        filepath = f"{current_dir}/../../data/processed"
        wandb.save(f"{filepath}/{TRAIN_PROMPT_FILENAME}.csv")
        wandb.save(f"{filepath}/{EVAL_PROMPT_FILENAME}.csv")


    wandb.finish()

    return model


