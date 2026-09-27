"""Fine-tuning multilingual language models for Urdu fake news classification.

This script supports vanilla and domain-adapted versions of XLM-R and mBERT
on the four Urdu fake news datasets used in the experiments.

Training is performed in two stages:
    1. The pretrained language model is frozen while the classification
       layers are trained.
    2. The entire model is unfrozen and fine-tuned using a lower
       learning rate.

The complete experiment is repeated over five random seeds, and the final
results are reported as mean ± standard deviation.
"""

import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel


# ============================================================
# Configuration
# ============================================================

# Dataset options:
#   ATG     -> Ax-to-Grind
#   UFN23   -> UrduFakeNews 2023
#   UFN21   -> UrduFakeNews 2021
#   UFake21 -> UrduFake 2021

DATASET_NAME = "ATG"


# Model examples:
#
# Vanilla XLM-R:
# MODEL_NAME = "xlm-roberta-base"
#
# Vanilla mBERT:
# MODEL_NAME = "google-bert/bert-base-multilingual-cased"
#
# Domain-adapted XLM-R:
# MODEL_NAME = "models/xlmr_urdu_news"
#
# Domain-adapted mBERT:
# MODEL_NAME = "models/mbert_urdu_news"

MODEL_NAME = "xlm-roberta-base"


DATASETS = {
    "ATG": {
        "path": "datasets/new_dataset1.xlsx",
        "max_length": 170,
    },
    "UFN23": {
        "path": "datasets/new_dataset2.xlsx",
        "max_length": 170,
    },
    "UFN21": {
        "path": "datasets/new_dataset3.xlsx",
        "max_length": 512,
    },
    "UFake21": {
        "path": "datasets/new_dataset4.xlsx",
        "max_length": 512,
    },
}


# Five random seeds used for repeated experiments.
SEEDS = [42, 99, 1005, 2023, 2024]

BATCH_SIZE = 32

FROZEN_EPOCHS = 20
UNFROZEN_EPOCHS = 20

FROZEN_LR = 1e-5
UNFROZEN_LR = 1e-6

DROPOUT_RATE = 0.5

OUTPUT_DIR = Path("outputs")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Set random seeds for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Dataset
# ============================================================

def load_dataset(dataset_name):
    """Load one of the preprocessed Urdu fake news datasets."""

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )

    dataset_config = DATASETS[dataset_name]
    dataset_path = Path(dataset_config["path"])

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    dataframe = pd.read_excel(dataset_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = (
        dataframe[["text", "label"]]
        .dropna()
        .reset_index(drop=True)
    )

    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    invalid_labels = set(
        dataframe["label"].unique()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            "Labels must be encoded as 0 or 1. "
            f"Found: {sorted(invalid_labels)}"
        )

    return dataframe, dataset_config["max_length"]


def split_dataset(dataframe, seed):
    """Create stratified 68/12/20 train/validation/test splits."""

    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=seed,
        stratify=dataframe["label"],
    )

    # 15% of the remaining 80% corresponds to 12% of the
    # complete dataset, leaving 68% for training.
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=0.15,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    return train_df, validation_df, test_df


# ============================================================
# Tokenization
# ============================================================

def tokenize_dataframe(
    dataframe,
    tokenizer,
    max_length,
):
    """Tokenize text and return model inputs and labels."""

    encodings = tokenizer(
        dataframe["text"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    inputs = {
        key: np.asarray(value)
        for key, value in encodings.items()
    }

    labels = dataframe[
        "label"
    ].to_numpy(dtype=np.float32)

    return inputs, labels


# ============================================================
# Classifier
# ============================================================

class FakeNewsClassifier(tf.keras.Model):
    """PLM encoder with the classification head used in the experiments."""

    def __init__(
        self,
        model_name,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.encoder = TFAutoModel.from_pretrained(
            model_name
        )

        self.dense1 = tf.keras.layers.Dense(
            256,
            activation="relu",
        )

        self.batch_norm1 = (
            tf.keras.layers.BatchNormalization()
        )

        self.dense2 = tf.keras.layers.Dense(
            128,
            activation="relu",
        )

        self.batch_norm2 = (
            tf.keras.layers.BatchNormalization()
        )

        self.dropout = tf.keras.layers.Dropout(
            dropout_rate
        )

        self.output_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )

    def call(self, inputs, training=False):

        outputs = self.encoder(
            inputs,
            training=training,
        )

        pooled_output = outputs[1]

        x = self.dense1(pooled_output)

        x = self.batch_norm1(
            x,
            training=training,
        )

        x = self.dense2(x)

        x = self.batch_norm2(
            x,
            training=training,
        )

        x = self.dropout(
            x,
            training=training,
        )

        return self.output_layer(x)


# ============================================================
# Training
# ============================================================

def compile_model(model, learning_rate):
    """Compile the model for binary classification."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            )
        ],
    )


def train_model(
    model,
    train_inputs,
    train_labels,
    validation_inputs,
    validation_labels,
    run_dir,
):
    """Perform frozen and unfrozen fine-tuning."""

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Stage 1: frozen PLM
    # --------------------------------------------------------

    print("\nStage 1: Frozen encoder")
    print(
        f"Learning rate: {FROZEN_LR} | "
        f"Epochs: {FROZEN_EPOCHS}"
    )

    model.encoder.trainable = False

    compile_model(
        model,
        FROZEN_LR,
    )

    frozen_checkpoint = (
        run_dir / "best_frozen.weights.h5"
    )

    frozen_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            str(frozen_checkpoint),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    model.fit(
        train_inputs,
        train_labels,
        validation_data=(
            validation_inputs,
            validation_labels,
        ),
        batch_size=BATCH_SIZE,
        epochs=FROZEN_EPOCHS,
        callbacks=[frozen_callback],
        verbose=1,
    )

    model.load_weights(
        str(frozen_checkpoint)
    )

    # --------------------------------------------------------
    # Stage 2: unfrozen PLM
    # --------------------------------------------------------

    print("\nStage 2: Unfrozen encoder")
    print(
        f"Learning rate: {UNFROZEN_LR} | "
        f"Epochs: {UNFROZEN_EPOCHS}"
    )

    model.encoder.trainable = True

    compile_model(
        model,
        UNFROZEN_LR,
    )

    unfrozen_checkpoint = (
        run_dir / "best_unfrozen.weights.h5"
    )

    unfrozen_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            str(unfrozen_checkpoint),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    model.fit(
        train_inputs,
        train_labels,
        validation_data=(
            validation_inputs,
            validation_labels,
        ),
        batch_size=BATCH_SIZE,
        epochs=UNFROZEN_EPOCHS,
        callbacks=[unfrozen_callback],
        verbose=1,
    )

    model.load_weights(
        str(unfrozen_checkpoint)
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    test_inputs,
    test_labels,
):
    """Evaluate the classifier on the held-out test set."""

    probabilities = model.predict(
        test_inputs,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            test_labels,
            predictions,
        ),
        "precision": precision_score(
            test_labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            test_labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            test_labels,
            predictions,
            zero_division=0,
        ),
    }


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    dataframe,
    tokenizer,
    max_length,
    seed,
):
    """Run one complete train/evaluate experiment."""

    print("\n" + "=" * 60)
    print(f"Seed: {seed}")
    print("=" * 60)

    # Remove models from the previous run before creating
    # a freshly initialized classifier.
    tf.keras.backend.clear_session()
    gc.collect()

    set_seed(seed)

    train_df, validation_df, test_df = (
        split_dataset(
            dataframe,
            seed,
        )
    )

    print(
        f"Train: {len(train_df):,} | "
        f"Validation: {len(validation_df):,} | "
        f"Test: {len(test_df):,}"
    )

    train_inputs, train_labels = (
        tokenize_dataframe(
            train_df,
            tokenizer,
            max_length,
        )
    )

    validation_inputs, validation_labels = (
        tokenize_dataframe(
            validation_df,
            tokenizer,
            max_length,
        )
    )

    test_inputs, test_labels = (
        tokenize_dataframe(
            test_df,
            tokenizer,
            max_length,
        )
    )

    model = FakeNewsClassifier(
        MODEL_NAME,
        dropout_rate=DROPOUT_RATE,
    )

    run_dir = (
        OUTPUT_DIR
        / DATASET_NAME
        / f"seed_{seed}"
    )

    model = train_model(
        model,
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        run_dir,
    )

    results = evaluate_model(
        model,
        test_inputs,
        test_labels,
    )

    print("\nResults")
    print("-" * 40)

    for metric, value in results.items():
        print(
            f"{metric.capitalize():10s}: "
            f"{value:.4f}"
        )

    return results


# ============================================================
# Result Summary
# ============================================================

def summarize_results(all_results):
    """Calculate mean and standard deviation across runs."""

    results_df = pd.DataFrame(
        all_results
    )

    means = results_df.mean()
    stds = results_df.std(ddof=1)

    print("\n" + "=" * 60)
    print("Final Results: Mean ± Standard Deviation")
    print("=" * 60)

    for metric in results_df.columns:
        print(
            f"{metric.capitalize():10s}: "
            f"{means[metric]:.4f} ± "
            f"{stds[metric]:.4f}"
        )

    return results_df, means, stds


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Urdu Fake News Classification")
    print("=" * 60)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Model:   {MODEL_NAME}")

    dataframe, max_length = load_dataset(
        DATASET_NAME
    )

    print(f"Samples: {len(dataframe):,}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Seeds: {SEEDS}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    all_results = []

    for seed in SEEDS:

        results = run_experiment(
            dataframe,
            tokenizer,
            max_length,
            seed,
        )

        results["seed"] = seed
        all_results.append(results)

    # Keep seed separate from the metrics when calculating
    # the mean and standard deviation.
    metric_results = [
        {
            key: value
            for key, value in result.items()
            if key != "seed"
        }
        for result in all_results
    ]

    results_df, means, stds = (
        summarize_results(
            metric_results
        )
    )

    # Save individual runs.
    results_df.insert(
        0,
        "seed",
        SEEDS,
    )

    results_dir = (
        OUTPUT_DIR
        / DATASET_NAME
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        results_dir / "results.csv",
        index=False,
    )

    # Save summary statistics.
    summary_df = pd.DataFrame({
        "metric": means.index,
        "mean": means.values,
        "std": stds.values,
    })

    summary_df.to_csv(
        results_dir / "summary.csv",
        index=False,
    )

    print(
        f"\nResults saved to: "
        f"{results_dir}"
    )


if __name__ == "__main__":
    main()
