import os
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy


def get_tokenizer(pretrain, model, padding_side="right", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, 
                                              use_fast=use_fast, add_bos_token=False)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.unk_token_id
    return tokenizer


def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    is_shuffle=True,
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                files = sorted(files)
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")
            else:
                strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data and len(data.keys()) == 1:
            # using probability to sample from train
            if probabilities[i] < 1.0:
                split_data = data['train'].train_test_split(train_size=probabilities[i], seed=seed)
                train_data_list.append(split_data['train'])
                eval_data_list.append(split_data['test'])
            else:
                train_data_list.append(data['train'])
        else:
            train_data_list.append(data['train'])
            eval_data_list.append(data['validation'])
            # train will contains eval? TODO

    # merge datasets
    if strategy.is_rank_0():
        print(f"train:{train_data_list}\neval:{eval_data_list}")

    train_dataset = concatenate_datasets(train_data_list)
    if is_shuffle:
        train_dataset = train_dataset.shuffle(seed=seed)
    if return_eval:
        eval_dataset = concatenate_datasets(eval_data_list)
        if is_shuffle:
            eval_dataset = eval_dataset.shuffle(seed=seed)
        return train_dataset, eval_dataset
    else:
        return train_dataset
