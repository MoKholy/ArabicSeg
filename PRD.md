# Baseline Evaluation PRD

## Goal
Evaluate segmentation baselines on `ArabicSeg/Notebooks/Data/4_data.tsv` for two tasks:

- `a -> c` (document reconstruction to sentence segmentation)
- `b -> c` (paragraph stream to sentence segmentation)

Where:

- `a`: document-level stream with no newlines
- `b`: paragraph-level stream separated by provided paragraph newlines
- `c`: linguistic sentence segmentation target

## Dataset and Grouping Rules

- Input file: `ArabicSeg/Notebooks/Data/4_data.tsv`
- Evaluate only rows where `new_split == dev`
- Group documents by `doc_name`
- Sort rows inside each document by:
  - `paragraph_id` (numeric)
  - `sent_id_in_paragraph` (numeric)
- Keep `source` per document for debugging reports
- Normalize sentence text before inference by removing literal `<SEG>` markers

## Task Definitions

## 1) Document Reconstruction (`a -> c`)

- For each document, concatenate all sentences across all paragraphs into one stream.
- Gold segmentation is sentence boundaries from the original sentence rows.
- Run baselines on the full stream and compare predicted boundaries against gold.
- Include an additional baseline: `paragraph_breaks_only`
  - Prediction is created only at paragraph boundaries (no within-paragraph sentence boundaries).

## 2) Semantic Segmentation (`b -> c`)

- For each paragraph in each document, concatenate paragraph sentences into one paragraph stream.
- Gold segmentation is sentence boundaries inside that paragraph.
- Run baselines paragraph-by-paragraph and aggregate metrics globally and per source.

## Baselines

- `sentencizer` (spaCy Arabic blank pipeline with sentencizer)
- `pysbd` (Arabic segmenter)
- `punkt` (NLTK Punkt sentence tokenizer trained on `new_split == train`)
  - Model is saved under `ArabicSeg/Notebooks/baselines/punkt_train.pkl` (folder auto-created).
- `ersatz` (optional, only when binary path is supplied)
- `paragraph_breaks_only` (only for `a -> c`)

## Metrics

- Precision, Recall, F1 using boundary-label evaluation:
  - Sentence with `n` tokens contributes `n-1` zeros and a trailing one.
- Compare flattened gold vs prediction labels with zero-division handling.
- Guardrails:
  - Raise explicit errors when gold/predicted label lengths mismatch.
  - Error message includes task and document/paragraph identifiers.

## Output Contract

Script: `ArabicSeg/Notebooks/get_baselines.py`

JSON output shape:

```json
{
  "split": "dev",
  "tasks": {
    "a_to_c": {
      "<baseline>": {
        "overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "per_source": {
          "<source>": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }
      }
    },
    "b_to_c": {
      "<baseline>": {
        "overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "per_source": {
          "<source>": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }
      }
    }
  }
}
```

## Usage

From project root:

`python3 ArabicSeg/Notebooks/get_baselines.py --split dev`

Optional arguments:

- `--data-path` custom TSV path
- `--ersatz-bin` path to ersatz binary (enables ersatz baseline)
- `--output-path` save report JSON to file
- `--output-tsv-path` save Excel-friendly TSV tables for overall and per-source metrics
- `--output-labels-dir` save detailed per-word prediction TSVs:
  - `a_to_c_predictions.tsv`
  - `b_to_c_predictions.tsv`

Detailed prediction TSV columns:

- `source`
- `doc_name`
- `paragraph_id`
- `sent_id_in_paragraph`
- `word`
- `true_label_for_word(0/1)`
- `<baseline>_pred` columns for each active baseline
