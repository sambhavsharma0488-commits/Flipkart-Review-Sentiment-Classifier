# Flipkart Review Sentiment Classification

Fine-tuned a pretrained transformer model to classify Flipkart product reviews into **positive**, **negative**, or **neutral** sentiment.

## Overview

- **Dataset**: Flipkart product reviews (Kaggle, ~205k rows) — `product_name`, `review_summary` used
- **Base model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa-base, 3-class sentiment)
- **Approach**: pseudo-labeling + manual fact-checking + fine-tuning + full-dataset inference

## Pipeline

1. **`1_Pseudo_review_generater`** — Sampled 5,000 reviews, generated pseudo-labels using the pretrained base model.
2. **Manual fact-checking** — Verified all 5,000 pseudo-labeled rows by hand (corrected ~11 rows).
3. **`2_Fine_tuning_model`** — Fine-tuned the base model on the 5,000 fact-checked rows (80/20 train/test split, 2 epochs).
4. **`3_sentimental_classification`** — Used the fine-tuned model to classify the full ~205k row dataset.

## Results

- **Test accuracy**: 96.8%
- **Train accuracy**: 98.6%
- Class-wise F1: negative 0.96, positive 0.98, neutral 0.87 (neutral had the least training data)

## Files

| File | Description |
|---|---|
| `1_Pseudo_review_generater.ipynb` | Generates pseudo-labels on a 5,000-row sample |
| `2_Fine_tuning_model.ipynb` | Fine-tunes the model on fact-checked labels |
| `3_sentimental_classification.ipynb` | Runs inference on the full dataset |
| `Review_unlabelled.csv` | Raw dataset before labeling |
| `Pseudo_review_label_checked.csv` | 5,000 manually fact-checked labeled rows |
| `Review_labelled.csv` | Final ~205k rows with predicted sentiment |

## Tools

Python, pandas, Hugging Face `transformers`, `datasets`, scikit-learn, Google Colab (GPU)

## Key Insight

Pseudo-labeling with a pretrained model before fine-tuning works well when paired with manual fact-checking on a representative sample — it kept manual labeling effort low (5k rows instead of 205k) while still reaching ~97% test accuracy.
