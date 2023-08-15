import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def get_question_prompts(
        train_prop: float = 0.95,
        shuffle: bool = True
) -> Tuple[List[str], List[str]]:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_file = current_dir + "/../../data/processed/TruthfulQA_questions.csv"
    question_data = pd.read_csv(data_file)

    indices = list(range(len(question_data)))
    if shuffle:
        np.random.shuffle(indices)

    train_split = int(np.floor(train_prop * len(question_data)))
    train_indices, eval_indices = indices[:train_split], indices[train_split:]

    train_prompts = [str(prompt) for prompt
                     in list(question_data["Question"][train_indices])]
    eval_prompts = [str(prompt) for prompt
                    in list(question_data["Question"][eval_indices])]

    return train_prompts, eval_prompts
