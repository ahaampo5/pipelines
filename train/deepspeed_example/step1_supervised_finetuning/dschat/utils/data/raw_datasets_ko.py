# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
from .raw_datasets import PromptRawDataset
import re



# English dataset
class HuggingFaceTB_SmoltalkDataset_KO(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mncai/foundation_model_smoltalk_ko_translate"
        self.dataset_name_clean = "HuggingFaceTB_SmoltalkDataset_KO"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["train"].select(range(10))

    def get_prompt(self, sample):
        return sample["prompt_mnc"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt_mnc"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt_mnc"] + ""

