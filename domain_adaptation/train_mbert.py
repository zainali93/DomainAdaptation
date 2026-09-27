"""Domain-adaptive pretraining of mBERT on the Urdu News Dataset 1M.

This script performs masked language model (MLM) fine-tuning of mBERT
on the Urdu News Dataset 1M for domain adaptation to Urdu news text.

The adapted model can subsequently be fine-tuned for downstream
Urdu fake news classification tasks.
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import tensorflow as tf
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TFAutoModelForMaskedLM,
    create_optimizer,
)


MODEL_NAME = "google-bert/bert-base-multilingual-cased"
TEXT_COLUMN = "News Text"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain adaptation of mBERT on the Urdu News Dataset 1M."
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("datasets/Urdu-News-Dataset-1M.csv"),
        help="Path to the Urdu News Dataset 1M CSV file.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/mbert_urdu_news"),
        help="Directory for saving the domain-adapted model.",
    )

    return parser.parse_args()


def load_dataset(data_path):
    """Load and validate the Urdu News Dataset 1M."""

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Download the Urdu News Dataset 1M and place the CSV at this "
            "location, or specify its path using --data_path."
        )

    try_encodings = [
        "utf-8",
        "iso-8859-1",
        "latin-1",
        "utf-16",
    ]

    dataframe = None

    for encoding in try_encodings:
        try:
            dataframe = pd.read_csv(
                data_path,
                encoding=encoding,
            )
            break
        except UnicodeDecodeError:
            continue

    if dataframe is None:
        raise RuntimeError(
            f"Unable to decode dataset: {data_path}"
        )

    if TEXT_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Expected column '{TEXT_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe.dropna(
        subset=[TEXT_COLUMN]
    ).reset_index(drop=True)

    return dataframe


def split_dataset(dataframe):
    """Shuffle the corpus and create 80/10/10 data splits."""

    dataframe = dataframe.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    train_end = int(len(dataframe) * 0.8)
    validation_end = int(len(dataframe) * 0.9)

    train_df = dataframe.iloc[:train_end]
    validation_df = dataframe.iloc[
        train_end:validation_end
    ]
    test_df = dataframe.iloc[validation_end:]

    return train_df, validation_df, test_df


def create_huggingface_dataset(
    train_df,
    validation_df,
    test_df,
):
    """Convert the pandas dataframes to a Hugging Face DatasetDict."""

    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": train_df[
                        TEXT_COLUMN
                    ].astype(str).tolist()
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "text": validation_df[
                        TEXT_COLUMN
                    ].astype(str).tolist()
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": test_df[
                        TEXT_COLUMN
                    ].astype(str).tolist()
                }
            ),
        }
    )


def tokenize_and_group(
    dataset,
    tokenizer,
    chunk_size=128,
):
    """Tokenize articles and group tokens into fixed-length sequences."""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=256,
            truncation=True,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    def group_texts(examples):
        concatenated_examples = {
            key: sum(examples[key], [])
            for key in examples
        }

        total_length = len(
            concatenated_examples["input_ids"]
        )

        total_length = (
            total_length // chunk_size
        ) * chunk_size

        result = {
            key: [
                values[i : i + chunk_size]
                for i in range(
                    0,
                    total_length,
                    chunk_size,
                )
            ]
            for key, values
            in concatenated_examples.items()
        }

        result["labels"] = result[
            "input_ids"
        ].copy()

        return result

    return tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=(
            f"Grouping tokens into "
            f"{chunk_size}-token sequences"
        ),
    )


def main():
    args = parse_args()

    # Reproducibility
    tf.keras.utils.set_random_seed(SEED)

    # Mixed-precision training
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16"
    )

    print(f"Loading dataset: {args.data_path}")

    dataframe = load_dataset(args.data_path)

    print(
        f"Loaded {len(dataframe):,} articles."
    )

    train_df, validation_df, test_df = split_dataset(
        dataframe
    )

    print(
        "Dataset split:\n"
        f"  Train:      {len(train_df):,}\n"
        f"  Validation: {len(validation_df):,}\n"
        f"  Test:       {len(test_df):,}"
    )

    dataset = create_huggingface_dataset(
        train_df,
        validation_df,
        test_df,
    )

    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    lm_datasets = tokenize_and_group(
        dataset,
        tokenizer,
        chunk_size=128,
    )

    print(f"Loading model: {MODEL_NAME}")

    model = TFAutoModelForMaskedLM.from_pretrained(
        MODEL_NAME
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        return_tensors="tf",
    )

    tf_train_dataset = model.prepare_tf_dataset(
        lm_datasets["train"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=16,
    )

    # The evaluation split follows the training setup used
    # for the domain-adaptation experiment.
    tf_eval_dataset = model.prepare_tf_dataset(
        lm_datasets["test"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=16,
    )

    num_train_steps = len(tf_train_dataset)

    optimizer, _ = create_optimizer(
        init_lr=1e-4,
        num_warmup_steps=1000,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )

    model.compile(
        optimizer=optimizer
    )

    print("Starting domain-adaptive pretraining...")

    model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        epochs=7,
    )

    print("Evaluating adapted model...")

    eval_loss = model.evaluate(
        tf_eval_dataset,
        verbose=1,
    )

    if isinstance(eval_loss, (list, tuple)):
        eval_loss = eval_loss[0]

    eval_loss = float(eval_loss)
    perplexity = math.exp(eval_loss)

    print(f"Evaluation loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        args.output_dir
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    print(
        f"Model and tokenizer saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
