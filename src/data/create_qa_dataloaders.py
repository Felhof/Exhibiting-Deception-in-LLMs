# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler

from constants import TRUE_LABEL_ID, FALSE_LABEL_ID

class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.data_len = len(data)

        self.tokenizer = tokenizer
        # self.yes_idx = tokenizer("Yes").input_ids[0]
        # self.no_idx = tokenizer("No").input_ids[0]
        self.true_idx = TRUE_LABEL_ID
        self.false_idx = FALSE_LABEL_ID

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        qa, label = self.data.iloc[idx]

        return qa, label


# Pads all examples in batch to same dimension
class PadCollate():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.true_idx = TRUE_LABEL_ID
        self.false_idx = FALSE_LABEL_ID

    def __call__(self, batch):
        qa, label = zip(*batch)

        # Pad input
        x = self.tokenizer(qa, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # Convert each label to yes/no token
        label = list(label)
        for idx, i in enumerate(label):
            if label[idx] == 1:
                label[idx] = self.true_idx
            else:
                label[idx] = self.false_idx

        return input_ids, attention_mask, torch.tensor(label)


def create_qa_dataloaders(input_filepath, tokenizer, train_prop, batch_size, shuffle, with_eos=True):
    """
    Returns two PyTorch Dataloaders for the dataset: 
    one for training and one for testing. 
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_filepath = current_dir + "/../../" + input_filepath
    data = pd.read_csv(input_filepath)
    if with_eos:
        data["Full"] += tokenizer.eos_token
    dataset = QADataset(data, tokenizer)

    # Create splits
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(dataset)))
    train_indices, test_indices = indices[:train_split], indices[train_split:]

    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollate(
        tokenizer), sampler=SubsetRandomSampler(train_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollate(
        tokenizer), sampler=SubsetRandomSampler(test_indices))

    return train_loader, test_loader


def balance_dataset(data, filter=None):
    # Step 1: Count the occurrences of each label (0 and 1)
    label_counts = data['Label'].value_counts()

    # Step 2: Find the difference in counts between the two labels
    count_diff = label_counts[0] - label_counts[1]

    # Step 3: Randomly drop rows with label 0 to match the count of label 1
    if count_diff > 0:
        if filter is None:
            df_to_drop = data[
                data['Label'] == 0
            ].sample(n=count_diff, random_state=42)
        else:
            df_to_drop = data[
                (data['Label'] == 0) & (~data['Prompt'].isin(filter['Prompt']))
            ].sample(n=count_diff, random_state=42)
        data_balanced = data.drop(df_to_drop.index)
    else:
        data_balanced = data.copy()

    # Now 'df_balanced' contains a balanced dataset with an equal number of 0s and 1s in the "label" column.

    # You can reset the index of the balanced DataFrame if needed
    data_balanced.reset_index(drop=True, inplace=True)

    return data_balanced


def create_augmented_dataloaders(
        tokenizer, 
        train_prop=0.8, 
        shuffled_prop=0.16, 
        batch_size=16,
        balanced=True,
        with_eos=True
    ):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    tqa_augmented = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_augmented.csv"
    )
    tqa_shuffled = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_shuffled.csv"
    )
    tqa_vanilla = pd.read_csv(
        current_dir + "/../../data/processed/TruthfulQA_labeled.csv"
    )
    if with_eos:
        tqa_augmented["Prompt"] += tokenizer.eos_token
        tqa_shuffled["Prompt"] += tokenizer.eos_token
        tqa_vanilla["Full"] += tokenizer.eos_token

    num_tqa_shuffled = int(len(tqa_shuffled) * shuffled_prop)
    num_tqa_shuffled_train = int(num_tqa_shuffled * train_prop)

    tqa_shuffled_sample = tqa_shuffled.sample(n=num_tqa_shuffled, random_state=42)
    tqa_shuffled_train = tqa_shuffled_sample.sample(n=num_tqa_shuffled_train, random_state=42)
    tqa_shuffled_test = tqa_shuffled_sample.drop(tqa_shuffled_train.index)
    tqa_vanilla_sample = tqa_vanilla.sample(n=int(len(tqa_vanilla) * 0.1), random_state=42)

    num_tqa_augmented_train = int(len(tqa_augmented) * train_prop)

    tqa_augmented_train = tqa_augmented.sample(n=num_tqa_augmented_train, random_state=42)
    tqa_augmented_test = tqa_augmented.drop(tqa_augmented_train.index)

    data_train = pd.concat([tqa_augmented_train, tqa_shuffled_train], ignore_index=True)
    data_test = pd.concat([tqa_augmented_test, tqa_shuffled_test], ignore_index=True)

    if balanced:
        data_train = balance_dataset(data_train, tqa_shuffled_train)
        data_test = balance_dataset(data_test, tqa_shuffled_test)

    train_dataset = QADataset(data_train, tokenizer)
    test_dataset = QADataset(data_test, tokenizer)
    shuffled_test_dataset = QADataset(tqa_shuffled_test, tokenizer)
    vanilla_dataset = QADataset(tqa_vanilla_sample, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=PadCollate(
        tokenizer), sampler=RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=PadCollate(
        tokenizer))
    shuffled_test_loader = DataLoader(shuffled_test_dataset, batch_size=batch_size, collate_fn=PadCollate(
        tokenizer))
    vanilla_test_loader = DataLoader(vanilla_dataset, batch_size=batch_size, collate_fn=PadCollate(
        tokenizer))

    return train_loader, test_loader, shuffled_test_loader, vanilla_test_loader


def create_babi_dataloaders(
    tokenizer, 
    batch_size=16,
    with_eos=True
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    babi_train = pd.read_csv(
        current_dir + "/../../data/processed/babi_data_small_train.csv"
    )
    babi_val = pd.read_csv(
        current_dir + "/../../data/processed/babi_data_small_val.csv"
    )

    if with_eos:
        babi_train['prompt'] += tokenizer.eos_token
        babi_val['prompt'] += tokenizer.eos_token

    babi_train = babi_train.drop(
        columns=['passage', 'question', 'answer', 'task_type']
    )

    babi_val_t1 = babi_val[babi_val['task_type'] == 1]
    babi_val_t2 = babi_val[babi_val['task_type'] == 2]
    babi_val_t3 = babi_val[babi_val['task_type'] == 3]
    babi_val_t4 = babi_val[babi_val['task_type'] == 4]

    babi_val = babi_val.drop(
        columns=['passage', 'question', 'answer', 'task_type']
    )
    babi_val_t1 = babi_val_t1.drop(
        columns=['passage', 'question', 'answer', 'task_type']
    )
    babi_val_t2 = babi_val_t2.drop(
        columns=['passage', 'question', 'answer', 'task_type']
    )
    babi_val_t3 = babi_val_t3.drop(
        columns=['passage', 'question', 'answer', 'task_type']
    )
    babi_val_t4 = babi_val_t4.drop(
        columns=['passage', 'question', 'answer', 'task_type']
    )

    train_dataset = QADataset(babi_train, tokenizer)
    val_dataset = QADataset(babi_val, tokenizer)
    val_dataset_t1 = QADataset(babi_val_t1, tokenizer)
    val_dataset_t2 = QADataset(babi_val_t2, tokenizer)
    val_dataset_t3 = QADataset(babi_val_t3, tokenizer)
    val_dataset_t4 = QADataset(babi_val_t4, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer)
    )
    val_loader_t1 = DataLoader(
        val_dataset_t1,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer)
    )
    val_loader_t2 = DataLoader(
        val_dataset_t2,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer)
    )
    val_loader_t3 = DataLoader(
        val_dataset_t3,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer)
    )
    val_loader_t4 = DataLoader(
        val_dataset_t4,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer)
    )

    val_loaders = [val_loader_t1, val_loader_t2, val_loader_t3, val_loader_t4]
    return train_loader, val_loader, val_loaders


def create_multirc_dataloaders(
    tokenizer,
    batch_size=16,
    with_eos=True,
    easy=True
):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    train_filename = "easy_mrc_train" if easy else "hard_mrc_train"
    val_filename = "easy_mrc_val" if easy else "hard_mrc_val"
    
    multirc_train = pd.read_csv(
        current_dir + f"/../../data/processed/{train_filename}.csv"
    )
    multirc_val = pd.read_csv(
        current_dir + f"/../../data/processed/{val_filename}.csv"
    )

    if with_eos:
        multirc_train['prompt'] += tokenizer.eos_token
        multirc_val['prompt'] += tokenizer.eos_token

    multirc_train = multirc_train.drop(
        columns=['passage', 'query_and_answer', 'evidences']
    )
    multirc_val = multirc_val.drop(
        columns=['passage', 'query_and_answer', 'evidences']
    )

    train_dataset = QADataset(multirc_train, tokenizer)
    val_dataset = QADataset(multirc_val, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer),
        sampler=RandomSampler(train_dataset)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=PadCollate(tokenizer)
    )

    return train_loader, val_loader
