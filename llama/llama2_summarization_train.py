import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
import datasets
from datasets import Dataset, load_dataset

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from prompts import TRAINING_SUMMARIZATION_PROMPT_v2, TRAINING_SUMMARIZATION_PROMPT, TRAINING_SUMMARIZATION_PROMPT_v3
from transformers.utils import logging


def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = TRAINING_SUMMARIZATION_PROMPT_v2

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue="Summarize this dialogue:\n{}\n".format(dialogue),
            summary=summary,
        )
        instructions.append(example)

    return instructions


def prepare_samsum_data():
    dataset = load_dataset("samsum")
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    dialogues = train_dataset["dialogue"]
    summaries = train_dataset["summary"]
    train_instructions = prepare_instructions(dialogues, summaries)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions})
    )

    return train_dataset


def prepare_mts_instructions(dialogues, section_texts):
    instructions = []

    prompt = TRAINING_SUMMARIZATION_PROMPT_v2
    # prompt = TRAINING_SUMMARIZATION_PROMPT
    for dialogue, section_text in zip(dialogues, section_texts):
        example = prompt.format(
            dialogue=dialogue,
            summary=section_text,
            # summary='Section header:{}\nSection text:{}'.format(section_header, section_text),
        )
        instructions.append(example)

    return instructions


def prepare_mtsdialog_data(args):
    dataset = load_dataset("csv", data_files={"train": args.train_file, "test": args.test_file})
    # dataset = load_dataset("csv", data_files={"train": args.train_file})
    train_dataset = dataset["train"]


    dialogues = train_dataset["dialogue"]
    # section_headers = train_dataset["section_header"]
    section_texts = train_dataset["section_text"]

    train_instructions = prepare_mts_instructions(dialogues, section_texts)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions})
    )

    val_dataset = dataset["test"]
    val_dialogues = val_dataset["dialogue"]
    val_section_texts = val_dataset["section_text"]
    val_instructions = prepare_mts_instructions(val_dialogues, val_section_texts)
    val_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": val_instructions})
    )

    return train_dataset, val_dataset

def prepare_aci_instructions(dialogues, section_texts):
    instructions = []

    prompt = TRAINING_SUMMARIZATION_PROMPT_v3
    for dialogue, section_text in zip(dialogues, section_texts):
        example = prompt.format(
            dialogue=dialogue,
            summary=section_text,
            # summary='Section header:{}\nSection text:{}'.format(section_header, section_text),
        )
        instructions.append(example)

    return instructions

def prepare_acibench_data(args):
    dataset = load_dataset("csv", data_files={"train": args.train_file})

    train_dataset = dataset["train"]
    # val_dataset = dataset["test"]

    dialogues = train_dataset["dialogue"]
    notes = train_dataset["note"]
    # encounter_id = train_dataset["encounter_id"]
    # dataset = train_dataset['dataset']

    train_instructions = prepare_aci_instructions(dialogues, notes)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions})
    )

    return train_dataset

def main(args):
    if args.dataset == "MTS_Dialogue":
        train_dataset, val_dataset = prepare_mtsdialog_data(args)
    elif args.dataset == "aci-bench":
        train_dataset = prepare_acibench_data(args)

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        use_flash_attention_2=args.use_flash_attention,
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)


    results_dir = f"experiments/{args.pkg_prefix}_ds-{args.dataset}_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}"

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        # per_device_train_batch_size=92 if args.use_flash_attention else 64,
        # per_device_train_batch_size=48 if args.use_flash_attention else 32,
        #per_device_train_batch_size=32 if args.use_flash_attention else 16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=int(args.per_device_train_batch_size/2),
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        # logging_steps=4,
        # save_steps=4,
        logging_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        evaluation_strategy='epoch',
        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )

    # max_seq_length = args.max_seq_length  # max sequence length for model and packing of the dataset
    max_seq_length = None

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")


    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg_prefix", default="summarization_llama3")
    parser.add_argument("--pretrained_ckpt", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--per_device_train_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--use_flash_attention", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", type=str,
                        default="MTS_Dialogue") #aci-bench
    parser.add_argument("--train_file",
                        default="/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv")
    parser.add_argument("--validation_file",
                        default="/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv")
    parser.add_argument("--test_file",
                        default="/home/data2/yongfeng/code3/MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv")

    args = parser.parse_args()
    main(args)
