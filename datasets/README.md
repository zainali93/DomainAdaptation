# Datasets

This directory contains the four Urdu fake news datasets used for downstream
classification experiments in the paper **"Fake News Classification in Urdu
through Domain Adaptation of Multilingual Language Models."**

The datasets have been preprocessed into a common format to simplify
reproduction of the classification experiments.

## Dataset Files

| File | Dataset | Description |
|---|---|---|
| `new_dataset1.xlsx` | ATG | Ax-to-Grind |
| `new_dataset2.xlsx` | UFN23 | UrduFakeNews 2023 |
| `new_dataset3.xlsx` | UFN21 | UrduFakeNews 2021 |
| `new_dataset4.xlsx` | UFake21 | UrduFake 2021 |

The first two datasets (ATG and UFN23) primarily contain short news texts,
whereas UFN21 and UFake21 contain longer news articles.

## Data Format

All four files follow the same structure:

| Column | Description |
|---|---|
| `text` | Urdu news text |
| `label` | Binary class label |

The labels are encoded as:

- `0` — Real news
- `1` — Fake news

This standardized format is used directly by the classification script in
this repository.

## Urdu News Dataset 1M

The large-scale Urdu news corpus used for domain-adaptive pretraining is not
redistributed in this repository.

The experiments use the **Urdu News Dataset 1M**, containing approximately
1.04 million Urdu news articles.

The dataset can be downloaded from Kaggle:

**[Urdu News Dataset](https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset)**

After downloading the dataset, place the CSV file at:

```text
datasets/Urdu-News-Dataset-1M.csv
