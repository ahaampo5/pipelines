from .raw_datasets import PromptRawDataset
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

def get_token_length(text):
    """Get the number of tokens in a text."""
    return len(tokenizer.encode(text, add_special_tokens=False))

class AceReasoningDataset(PromptRawDataset):
    """
    Ace Reasoning dataset for supervised fine-tuning.
    This dataset is used for training models on reasoning tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "nvidia/AceReason-1.1-SFT"
        self.dataset_name_clean = "nvidia_AceReason-1.1-SFT"
        
    def get_train_data(self):
        return self.raw_datasets["train"].select(range(1000))

    def get_eval_data(self):
        return self.raw_datasets["train"].select(range(100))

    def get_prompt(self, sample):
        return sample["input"]

    def get_chosen(self, sample):
        return sample["output"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["input"] + sample["output"]

    def get_prompt_and_rejected(self, sample):
        return sample["input"] + ""
    
# FreedomIntelligence/medical-o1-reasoning-SFT
class MedicalReasoningDataset(PromptRawDataset):
    """
    Medical Reasoning dataset for supervised fine-tuning.
    This dataset is used for training models on medical reasoning tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "FreedomIntelligence/medical-o1-reasoning-SFT"
        self.dataset_name_clean = "FreedomIntelligence_medical-o1-reasoning-SFT"
        
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["train"].select(range(100))

    def get_prompt(self, sample):
        return sample["Question"]

    def get_chosen(self, sample):
        return sample["Complex_CoT"] + sample["Response"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["Question"] + sample["Complex_CoT"] + sample["Response"]

    def get_prompt_and_rejected(self, sample):
        return sample["Question"] + ""
    

###################################### Reasoning 360 Datasets ######################################
# Code #
class LeetcodeDataset(PromptRawDataset): # TODO: 테스트 케이스만 있고 솔루션은 없어서 GRPO 전용으로 보임 - 생성해야할듯
    """
    Leetcode dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/data/train/codegen__leetcode2k_1.3k.parquet"
        self.dataset_name_clean = "codegen_leetcode2k-1.3k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["query"]

    def get_chosen(self, sample):
        return sample["response"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["query"] + sample["response"]

    def get_prompt_and_rejected(self, sample):
        return sample["query"] + ""
    

class LivecondebenchDataset(PromptRawDataset):
    """
    Livecondebench dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/codegen__livecodebench_440.parquet"
        self.dataset_name_clean = "codegen_livecodebench"
        df_test = pd.read_parquet("/root/workspace/DeepSpeedExamples/raw_data/offline_eval/codegen__livecodebench_279.parquet")
        df_test = df_test.drop(columns=["tests", "reward_model"])
        self.raw_test_datasets = Dataset.from_pandas(df_test, preserve_index=False)
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_test_datasets

    def get_prompt(self, sample):
        return sample["query"]

    def get_chosen(self, sample):
        return sample["response"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["query"] + sample["response"]

    def get_prompt_and_rejected(self, sample):
        return sample["query"] + ""
    

class PrimeintellectDataset(PromptRawDataset):
    """
    Primeintellect dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        df = pd.read_parquet(dataset_name)
        dataset = Dataset.from_pandas(df, preserve_index=False)
        map_dataset = dataset.map(lambda x: {"input_length": get_token_length(x["problem"]), "output_length": get_token_length(x["solutions"][0])})
        map_dataset = map_dataset.map(lambda x: {"total_length": x["input_length"] + x["output_length"]})
        self.raw_datasets = map_dataset.filter(lambda x: x["total_length"] < 16_384)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/codegen__primeintellect_7.5k.parquet"
        self.dataset_name_clean = "codegen_primeintellect-7.5k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["problem"]

    def get_chosen(self, sample):
        return sample["solutions"][0]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["problem"] + sample["solutions"][0]

    def get_prompt_and_rejected(self, sample):
        return sample["problem"] + ""
    

class TacoDataset(PromptRawDataset):
    """
    Taco dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/codegen__taco_8.8k.parquet"
        self.dataset_name_clean = "codegen_taco-8.8k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return sample["solutions"][0]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + sample["solutions"][0]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""

class HumanevalDataset(PromptRawDataset):
    """
    Humaneval dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/offline_eval/codegen__humaneval_164.parquet"
        self.dataset_name_clean = "codegen_humaneval-1.6k"
        
    def get_train_data(self):
        return self.raw_datasets.select(range(0))

    def get_eval_data(self):
        return self.raw_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return sample["canonical_solution"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + sample["canonical_solution"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""


class MbppDataset(PromptRawDataset):
    """
    Mbpp dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/offline_eval/codegen__mbpp_500.parquet"
        self.dataset_name_clean = "codegen_mbpp-500"
        
    def get_train_data(self):
        return self.raw_datasets.select(range(0))

    def get_eval_data(self):
        return self.raw_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return sample["code"]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + sample["code"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    
# Logic #
class Arcagi1Dataset(PromptRawDataset):
    """
    Agi1 dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/logic__arcagi1_111.parquet"
        self.dataset_name_clean = "logic_arcagi1-111"
        df_test = pd.read_parquet("/root/workspace/DeepSpeedExamples/raw_data/offline_eval/logic__arcagi1_400.parquet")
        self.raw_test_datasets = Dataset.from_pandas(df_test, preserve_index=False)
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_test_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class Arcagi2Dataset(PromptRawDataset):
    """
    Agi1 dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/logic__arcagi2_190.parquet"
        self.dataset_name_clean = "logic_arcagi1-111"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class BarcDataset(PromptRawDataset):
    """
    BARC dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/logic__barc_1.6k.parquet"
        self.dataset_name_clean = "logic_barc-1.6k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class GraphLogicalDataset(PromptRawDataset):
    """
    Graph Logical dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/logic__graph_logical_1.2k.parquet"
        self.dataset_name_clean = "logic_graph_logical-1.2k"

    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class OrderingPuzzleDataset(PromptRawDataset):
    """
    Ordering Puzzle dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/logic__ordering_puzzle_1.9k.parquet"
        self.dataset_name_clean = "logic_ordering_puzzle-1.9k"

    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class ZebraPuzzleDataset(PromptRawDataset):
    """
    Zebra Puzzle dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/logic__zebra_puzzle_1.3k.parquet"
        self.dataset_name_clean = "logic_zebra_puzzle-1.3k"
        df_test = pd.read_parquet("/root/workspace/DeepSpeedExamples/raw_data/offline_eval/logic__zebra_puzzle_dataset_200.parquet")
        self.raw_test_datasets = Dataset.from_pandas(df_test, preserve_index=False)

    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_test_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "<answer> " + str([a for a in sample['reward_model']['ground_truth']]) + " </answer>"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

# Math #
class MathCombinedDataset(PromptRawDataset):
    """
    Math dataset for supervised fine-tuning.
    This dataset is used for training models on math tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/math__combined_54.4k.parquet"
        self.dataset_name_clean = "math_math_dataset-54.4k"

    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    
class Math500Dataset(PromptRawDataset):
    """
    Math dataset for supervised fine-tuning.
    This dataset is used for training models on math tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/offline_eval/math__math_500.parquet"
        self.dataset_name_clean = "math_math_dataset-500"

    def get_train_data(self):
        return self.raw_datasets.select(range(0))

    def get_eval_data(self):
        return self.raw_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class AimeRepeated8x240Dataset(PromptRawDataset):
    """
    Math AIME dataset for supervised fine-tuning.
    This dataset is used for training models on math tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/offline_eval/math__aime_repeated_8x_240.parquet"
        self.dataset_name_clean = "math_aime-1.2k"

    def get_train_data(self):
        return self.raw_datasets.select(range(0))

    def get_eval_data(self):
        return self.raw_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

# Simulation #
class CodeioDataset(PromptRawDataset):
    """
    CodeIO dataset for supervised fine-tuning.
    This dataset is used for training models on coding tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/simulation__codeio_3.7k.parquet"
        self.dataset_name_clean = "codeio_codeio-3.7k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return # sample["response"] # NOTE: CodeIO dataset does not have answer

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] # + sample["response"]

    def get_prompt_and_rejected(self, sample):
        return sample["query"] + ""
    
# STEM #
class WebDataset(PromptRawDataset):
    """
    Web dataset for supervised fine-tuning.
    This dataset is used for training models on STEM tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/stem__web_3.6k.parquet"
        self.dataset_name_clean = "stem_web-3.6k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""


class GPQADiamondDataset(PromptRawDataset):
    """
    GPQA dataset for supervised fine-tuning.
    This dataset is used for training models on STEM tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/offline_eval/stem__gpqa_diamond_198.parquet"
        self.dataset_name_clean = "stem_gpqa-1.2k"
        
    def get_train_data(self):
        return self.raw_datasets.select(range(0))

    def get_eval_data(self):
        return self.raw_datasets

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class SuperGPQADataset(PromptRawDataset):
    """
    Super GPQA dataset for supervised fine-tuning.
    This dataset is used for training models on STEM tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/offline_eval/stem__supergpqa_1k.parquet"
        self.dataset_name_clean = "stem_super_gpqa-1k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + sample['reward_model']['ground_truth'] + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""

# Table #
class HitabDataset(PromptRawDataset):
    """
    HITAB dataset for supervised fine-tuning.
    This dataset is used for training models on table tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/table__hitab_4.3k.parquet"
        self.dataset_name_clean = "table_hitab-4.3k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + str([a for a in sample['reward_model']['ground_truth']]) + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + str([a for a in sample['reward_model']['ground_truth']]) + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

class MultiHierDataset(PromptRawDataset):
    """
    MultiHier dataset for supervised fine-tuning.
    This dataset is used for training models on table tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "/root/workspace/DeepSpeedExamples/raw_data/train/table__multihier_1.5k.parquet"
        self.dataset_name_clean = "table_multihier-1.5k"
        
    def get_train_data(self):
        return self.raw_datasets

    def get_eval_data(self):
        return self.raw_datasets.select(range(0))

    def get_prompt(self, sample):
        return sample["prompt"][0]['content']

    def get_chosen(self, sample):
        return "$$\n\\boxed{" + str([a for a in sample['reward_model']['ground_truth']]) + "}$$"

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"][0]['content'] + "$$\n\\boxed{" + str([a for a in sample['reward_model']['ground_truth']]) + "}$$"

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"][0]['content'] + ""
    

# General #
class HuggingFaceTB_SmoltalkDataset(PromptRawDataset):
    """
    HuggingFaceTB Smoltalk dataset for supervised fine-tuning.
    This dataset is used for training models on general tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = "HuggingFaceTB/smoltalk"
        self.dataset_name_clean = "HuggingFaceTB_smoltalk"
        
    def get_train_data(self):
        return self.raw_datasets['train'].select(range(min([10000, len(self.raw_datasets['train'])])))

    def get_eval_data(self):
        return self.raw_datasets['test'].select(range(100))

    def get_prompt(self, sample):
        return sample["messages"][:-1]

    def get_chosen(self, sample):
        return sample["messages"][-1:]

    def get_rejected(self, sample):
        return ""

    def get_prompt_and_chosen(self, sample):
        return sample["messages"]

    def get_prompt_and_rejected(self, sample):
        return sample["messages"] + ""


class BaseConversationDataset(PromptRawDataset):
    """
    Base Conversation dataset for supervised fine-tuning.
    This dataset is used for training models on conversation tasks.
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, subset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, subset_name)
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name.replace("/", "_").replace("-", "_").replace(".", "_")
        
    def get_train_data(self):
        return self.raw_datasets['train'].select(range(min([1000, len(self.raw_datasets['train'])])))

    def get_eval_data(self):
        return self.raw_datasets['train'].select(range(10))

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
    

class HuggingFaceTB_SmoltalkDataset_EN(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mncai/foundation_model_smoltalk_en"
        self.dataset_name_clean = "HuggingFaceTB_SmoltalkDataset_EN"

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