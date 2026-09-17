# Text Mining of Mental Health Conditions

Fine-tuning **ModernBERT-base** to classify Reddit posts into six mental health categories. The project covers the full NLP pipeline: exploratory data analysis, text cleaning, tokenisation, dataset construction and transformer fine-tuning with Hugging Face.

---

## Problem

Given the title and body of a Reddit post, predict which mental health condition the post relates to. This is a **single-label, six-class classification** task.

| `class_id` | `class_name` |
|---:|---|
| 0 | ADHD |
| 1 | Anxiety |
| 2 | Bipolar |
| 3 | Depression |
| 4 | PTSD |
| 5 | None |

---

## Data

| Split | Rows |
|---|---:|
| Train | 13,727 |
| Validation | 1,488 |
| Test | 1,488 |

Columns: `ID`, `title`, `post`, `class_id`, `class_name`.

**EDA findings**
- No missing values in any split.
- Classes are approximately balanced across all six categories.
- `ID` and `class_name` are redundant for training (`class_id` already encodes the label).
- `title` and `post` are concatenated into a single `text` field for the model.

---

## Preprocessing

1. **Concatenate** `title + " " + post` → `text`.
2. **Drop** redundant columns (`ID`, `class_name`, `title`, `post`).
3. **Strip HTML tags** with a regex substitution.
4. **Flatten Reddit markdown links** — `[text](url)` → `text`.
5. **Anonymise usernames** — `u/username` → `USER`, `@mention` → `@USER`.
6. **Whitespace normalisation** — collapse runs of whitespace and trim.

**Punctuation is deliberately retained.** A frequency count showed the dominant special characters are ordinary punctuation (`.` `,` `"` `'` `(` `)` `:` `!` `?`), which carries contextual and semantic signal for a transformer. Rarer symbols were not treated as noise without evidence, so no character stripping was applied.

---

## Modelling

| Component | Choice |
|---|---|
| Base model | `answerdotai/ModernBERT-base` |
| Head | `AutoModelForSequenceClassification`, `num_labels=6` |
| Tokeniser | ModernBERT tokeniser, `truncation=True`, no fixed padding |
| Batching | `DataCollatorWithPadding` — dynamic padding to the longest sequence per batch |
| Dataset format | Hugging Face `Dataset` objects built from tokenised dicts |

Padding is applied at batch time rather than at tokenisation time, which avoids padding every sequence to the corpus maximum and reduces wasted computation.

### Training configuration

```python
TrainingArguments(
    output_dir="./modernbert_mhc",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)
```

Model selection uses **F1** on the validation split, with the best checkpoint reloaded at the end of training.

---

## Results

| Split | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Validation | — | — | — |
| Test | — | — | — |

<!-- Fill in after the training run; add the per-class classification report and confusion matrix below. -->

---

## Repository structure

```
.
├── text-mining-mental-health-conditions.ipynb   # end-to-end pipeline
├── README.md
└── data/                                        # not tracked
    ├── both_train.csv
    ├── both_val.csv
    └── both_test.csv
```

---

## Running the notebook

```bash
pip install pandas numpy torch transformers datasets scikit-learn
```

The notebook was developed on Kaggle with GPU acceleration. Update the three path variables near the top to point at your own copy of the data:

```python
train_file_path = ".../both_train.csv"
val_file_path   = ".../both_val.csv"
test_file_path  = ".../both_test.csv"
```

A GPU is strongly recommended — fine-tuning ModernBERT-base on ~14k posts for 3 epochs is impractical on CPU.

---

## Roadmap

- [ ] Word clouds per category
- [ ] Complete training run and log evaluation metrics
- [ ] Per-class classification report and confusion matrix
- [ ] Error analysis, with attention to Depression / Anxiety / Bipolar confusions
- [ ] Baseline comparison (TF-IDF + linear model) to quantify the transformer's gain
- [ ] Inference script or demo

---

## Ethical note

The dataset consists of public Reddit posts about mental health. Usernames and mentions are masked during preprocessing. This model is a research artefact for text classification only — it is **not** a diagnostic tool and must not be used to infer the mental health status of any individual.
