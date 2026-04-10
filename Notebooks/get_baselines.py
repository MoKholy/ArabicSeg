import argparse
import json
import os
import pickle
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    import spacy
except ImportError:
    spacy = None

try:
    import pysbd
except ImportError:
    pysbd = None

try:
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except ImportError:
    PunktSentenceTokenizer = None


@dataclass
class Sentence:
    sent_id_in_paragraph: int
    text: str


@dataclass
class Paragraph:
    paragraph_id: int
    sentences: List[Sentence]


@dataclass
class Document:
    doc_name: str
    source: str
    paragraphs: List[Paragraph]


def normalize_sentence(text: str) -> str:
    cleaned = text.replace(" <SEG> ", " ")
    return " ".join(cleaned.split())


def get_labels(sentences: List[str]) -> List[int]:
    labels: List[int] = []
    for sentence in sentences:
        token_count = len(sentence.split())
        if token_count <= 0:
            continue
        if token_count > 1:
            labels.extend([0] * (token_count - 1))
        labels.append(1)
    return labels


def evaluate(gold_labels: List[List[int]], pred_labels: List[List[int]]) -> Dict[str, float]:
    all_gold = [label for unit in gold_labels for label in unit]
    all_pred = [label for unit in pred_labels for label in unit]
    if len(all_gold) != len(all_pred):
        raise ValueError(
            f"Length mismatch while evaluating labels: gold={len(all_gold)} pred={len(all_pred)}"
        )
    return {
        "precision": precision_score(y_true=all_gold, y_pred=all_pred, zero_division=0),
        "recall": recall_score(y_true=all_gold, y_pred=all_pred, zero_division=0),
        "f1": f1_score(y_true=all_gold, y_pred=all_pred, zero_division=0),
    }


def load_documents(path: str, split: str = "dev") -> List[Document]:
    df = pd.read_csv(path, sep="\t")
    df = df[df["new_split"] == split].copy()
    if df.empty:
        return []
    df["paragraph_id"] = pd.to_numeric(df["paragraph_id"])
    df["sent_id_in_paragraph"] = pd.to_numeric(df["sent_id_in_paragraph"])
    df = df.sort_values(["doc_name", "paragraph_id", "sent_id_in_paragraph"])

    documents: List[Document] = []
    for doc_name, doc_df in df.groupby("doc_name", sort=False):
        source_values = doc_df["source"].dropna().astype(str).unique().tolist()
        if len(source_values) != 1:
            raise ValueError(f"Document {doc_name} has multiple sources: {source_values}")
        source = source_values[0]

        paragraphs: List[Paragraph] = []
        for paragraph_id, paragraph_df in doc_df.groupby("paragraph_id", sort=True):
            sentences: List[Sentence] = []
            for _, row in paragraph_df.iterrows():
                sentences.append(
                    Sentence(
                        sent_id_in_paragraph=int(row["sent_id_in_paragraph"]),
                        text=normalize_sentence(str(row["text"])),
                    )
                )
            paragraphs.append(Paragraph(paragraph_id=int(paragraph_id), sentences=sentences))

        documents.append(Document(doc_name=doc_name, source=source, paragraphs=paragraphs))

    documents.sort(key=lambda doc: doc.doc_name)
    return documents


def build_punkt_training_corpus(documents: List[Document]) -> str:
    corpus_parts: List[str] = []
    for doc in documents:
        for paragraph in doc.paragraphs:
            for sentence in paragraph.sentences:
                sentence_text = sentence.text.strip()
                if sentence_text:
                    corpus_parts.append(sentence_text)
            corpus_parts.append("")
        corpus_parts.append("")
    return "\n".join(corpus_parts)


def make_sentencizer_runner() -> Callable[[str], List[str]]:
    if spacy is None:
        raise RuntimeError("spacy is not installed. Install it to run sentencizer baseline.")
    nlp = spacy.blank("ar")
    nlp.add_pipe("sentencizer")

    def _run(doc: str) -> List[str]:
        return [str(s).strip() for s in nlp(doc).sents if str(s).strip()]

    return _run


def make_pysbd_runner() -> Callable[[str], List[str]]:
    if pysbd is None:
        raise RuntimeError("pysbd is not installed. Install it to run pysbd baseline.")
    seg = pysbd.Segmenter(language="ar", clean=False)

    def _run(doc: str) -> List[str]:
        return [s.strip() for s in seg.segment(doc) if s.strip()]

    return _run


def make_punkt_runner(
    train_documents: List[Document], model_path: Path
) -> Callable[[str], List[str]]:
    if PunktSentenceTokenizer is None:
        raise RuntimeError("nltk punkt is not installed. Install nltk to run punkt baseline.")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        with open(model_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        training_corpus = build_punkt_training_corpus(train_documents)
        if not training_corpus.strip():
            raise RuntimeError("No training text found for punkt training split.")
        tokenizer = PunktSentenceTokenizer(train_text=training_corpus)
        with open(model_path, "wb") as f:
            pickle.dump(tokenizer, f)

    def _run(doc: str) -> List[str]:
        return [s.strip() for s in tokenizer.tokenize(doc) if s.strip()]

    return _run


def make_ersatz_runner(ersatz_bin: str) -> Callable[[str], List[str]]:
    def _run(doc: str) -> List[str]:
        res = subprocess.run(
            [ersatz_bin],
            input=doc,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"ersatz failed with code {res.returncode}: {res.stderr.strip()}")
        return [seg.strip() for seg in res.stdout.split("\n") if seg.strip()]

    return _run


def build_baselines(
    train_documents: List[Document], ersatz_bin: str = ""
) -> Dict[str, Callable[[str], List[str]]]:
    baselines: Dict[str, Callable[[str], List[str]]] = {}
    static_factories = (
        ("sentencizer", make_sentencizer_runner),
        ("pysbd", make_pysbd_runner),
    )
    for name, factory in static_factories:
        try:
            baselines[name] = factory()
        except Exception as exc:
            print(f"[warn] baseline '{name}' disabled: {exc}")
    try:
        script_dir = Path(__file__).resolve().parent
        punkt_model_path = script_dir / "baselines" / "punkt_train.pkl"
        baselines["punkt"] = make_punkt_runner(
            train_documents=train_documents, model_path=punkt_model_path
        )
    except Exception as exc:
        print(f"[warn] baseline 'punkt' disabled: {exc}")
    if ersatz_bin:
        baselines["ersatz"] = make_ersatz_runner(ersatz_bin)
    return baselines


def flatten_document_word_rows(doc: Document) -> Tuple[List[Dict[str, object]], List[str], List[int]]:
    rows: List[Dict[str, object]] = []
    gold_segments: List[str] = []
    gold_labels: List[int] = []

    for paragraph in doc.paragraphs:
        for sentence in paragraph.sentences:
            words = sentence.text.split()
            if not words:
                continue
            gold_segments.append(sentence.text)
            for idx, word in enumerate(words):
                true_label = 1 if idx == len(words) - 1 else 0
                rows.append(
                    {
                        "source": doc.source,
                        "doc_name": doc.doc_name,
                        "paragraph_id": paragraph.paragraph_id,
                        "sent_id_in_paragraph": sentence.sent_id_in_paragraph,
                        "word": word,
                        "true_label_for_word(0/1)": true_label,
                    }
                )
                gold_labels.append(true_label)
    return rows, gold_segments, gold_labels


def flatten_paragraph_word_rows(
    doc: Document, paragraph: Paragraph
) -> Tuple[List[Dict[str, object]], List[str], List[int]]:
    rows: List[Dict[str, object]] = []
    gold_segments: List[str] = []
    gold_labels: List[int] = []
    for sentence in paragraph.sentences:
        words = sentence.text.split()
        if not words:
            continue
        gold_segments.append(sentence.text)
        for idx, word in enumerate(words):
            true_label = 1 if idx == len(words) - 1 else 0
            rows.append(
                {
                    "source": doc.source,
                    "doc_name": doc.doc_name,
                    "paragraph_id": paragraph.paragraph_id,
                    "sent_id_in_paragraph": sentence.sent_id_in_paragraph,
                    "word": word,
                    "true_label_for_word(0/1)": true_label,
                }
            )
            gold_labels.append(true_label)
    return rows, gold_segments, gold_labels


def evaluate_a_to_c(
    documents: List[Document], baselines: Dict[str, Callable[[str], List[str]]]
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], pd.DataFrame]:
    labels_by_baseline: Dict[str, List[List[int]]] = defaultdict(list)
    labels_by_baseline_source: Dict[str, Dict[str, List[List[int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gold_all: List[List[int]] = []
    gold_by_source: Dict[str, List[List[int]]] = defaultdict(list)
    prediction_rows: List[Dict[str, object]] = []

    for doc in documents:
        doc_rows, gold_segments, gold_labels = flatten_document_word_rows(doc)
        doc_text = " ".join(gold_segments)
        gold_all.append(gold_labels)
        gold_by_source[doc.source].append(gold_labels)
        doc_predictions: Dict[str, List[int]] = {}

        for baseline_name, baseline_fn in baselines.items():
            pred_segments = baseline_fn(doc_text)
            pred_labels = get_labels(pred_segments)
            if len(pred_labels) != len(gold_labels):
                raise ValueError(
                    f"a_to_c mismatch for doc={doc.doc_name} baseline={baseline_name}: "
                    f"gold={len(gold_labels)} pred={len(pred_labels)}"
                )
            labels_by_baseline[baseline_name].append(pred_labels)
            labels_by_baseline_source[baseline_name][doc.source].append(pred_labels)
            doc_predictions[f"{baseline_name}_pred"] = pred_labels

        paragraph_break_segments = [
            " ".join(sentence.text for sentence in p.sentences).strip() for p in doc.paragraphs if p.sentences
        ]
        para_pred_labels = get_labels(paragraph_break_segments)
        if len(para_pred_labels) != len(gold_labels):
            raise ValueError(
                f"a_to_c mismatch for doc={doc.doc_name} baseline=paragraph_breaks_only: "
                f"gold={len(gold_labels)} pred={len(para_pred_labels)}"
            )
        labels_by_baseline["paragraph_breaks_only"].append(para_pred_labels)
        labels_by_baseline_source["paragraph_breaks_only"][doc.source].append(para_pred_labels)
        doc_predictions["paragraph_breaks_only_pred"] = para_pred_labels

        for idx, row in enumerate(doc_rows):
            output_row = dict(row)
            for pred_col, pred_values in doc_predictions.items():
                output_row[pred_col] = pred_values[idx]
            prediction_rows.append(output_row)

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for baseline_name, preds in labels_by_baseline.items():
        metrics[baseline_name] = {
            "overall": evaluate(gold_all, preds),
            "per_source": {
                source: evaluate(gold_by_source[source], labels_by_baseline_source[baseline_name][source])
                for source in sorted(gold_by_source.keys())
            },
        }
    predictions_df = pd.DataFrame(prediction_rows)
    return metrics, predictions_df


def evaluate_b_to_c(
    documents: List[Document], baselines: Dict[str, Callable[[str], List[str]]]
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], pd.DataFrame]:
    labels_by_baseline: Dict[str, List[List[int]]] = defaultdict(list)
    labels_by_baseline_source: Dict[str, Dict[str, List[List[int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gold_all: List[List[int]] = []
    gold_by_source: Dict[str, List[List[int]]] = defaultdict(list)
    prediction_rows: List[Dict[str, object]] = []

    for doc in documents:
        for paragraph in doc.paragraphs:
            para_rows, paragraph_segments, gold_labels = flatten_paragraph_word_rows(doc, paragraph)
            paragraph_text = " ".join(paragraph_segments)
            gold_all.append(gold_labels)
            gold_by_source[doc.source].append(gold_labels)
            para_predictions: Dict[str, List[int]] = {}

            for baseline_name, baseline_fn in baselines.items():
                pred_segments = baseline_fn(paragraph_text)
                pred_labels = get_labels(pred_segments)
                if len(pred_labels) != len(gold_labels):
                    raise ValueError(
                        f"b_to_c mismatch for doc={doc.doc_name} paragraph={paragraph.paragraph_id} "
                        f"baseline={baseline_name}: gold={len(gold_labels)} pred={len(pred_labels)}"
                    )
                labels_by_baseline[baseline_name].append(pred_labels)
                labels_by_baseline_source[baseline_name][doc.source].append(pred_labels)
                para_predictions[f"{baseline_name}_pred"] = pred_labels

            # Labels-only helper baseline for TSV debugging output:
            # predict a single segment per paragraph (no internal breaks).
            paragraph_only_pred_labels = get_labels([paragraph_text]) if paragraph_text.strip() else []
            if len(paragraph_only_pred_labels) != len(gold_labels):
                raise ValueError(
                    f"b_to_c mismatch for doc={doc.doc_name} paragraph={paragraph.paragraph_id} "
                    f"baseline=paragraph_breaks_only: gold={len(gold_labels)} "
                    f"pred={len(paragraph_only_pred_labels)}"
                )
            para_predictions["paragraph_breaks_only_pred"] = paragraph_only_pred_labels

            for idx, row in enumerate(para_rows):
                output_row = dict(row)
                for pred_col, pred_values in para_predictions.items():
                    output_row[pred_col] = pred_values[idx]
                prediction_rows.append(output_row)

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for baseline_name, preds in labels_by_baseline.items():
        metrics[baseline_name] = {
            "overall": evaluate(gold_all, preds),
            "per_source": {
                source: evaluate(gold_by_source[source], labels_by_baseline_source[baseline_name][source])
                for source in sorted(gold_by_source.keys())
            },
        }
    predictions_df = pd.DataFrame(prediction_rows)
    return metrics, predictions_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation baselines on 4_data.tsv")
    parser.add_argument(
        "--data-path",
        default="/Users/mohammed.elkholy/Desktop/Projects/ArabicSeg/Notebooks/Data/4_data.tsv",
        help="Path to TSV input data.",
    )
    parser.add_argument("--split", default="dev", help="Value of `new_split` to evaluate.")
    parser.add_argument(
        "--ersatz-bin",
        default="",
        help="Optional path to ersatz binary. If not provided, ersatz baseline is skipped.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional output JSON path. If not set, report is printed to stdout only.",
    )
    parser.add_argument(
        "--output-tsv-path",
        default="",
        help="Optional output TSV path for Excel-friendly tables (overall + per_source).",
    )
    parser.add_argument(
        "--output-labels-dir",
        default="",
        help="Optional directory to write per-word label/prediction TSVs for a_to_c and b_to_c.",
    )
    return parser.parse_args()


def write_metrics_tsv(report: Dict[str, object], output_tsv_path: str) -> None:
    tasks = report.get("tasks", {})
    overall_rows: List[Dict[str, object]] = []
    per_source_rows: List[Dict[str, object]] = []

    for task_name in ("a_to_c", "b_to_c"):
        task_metrics = tasks.get(task_name, {})
        for baseline_name in sorted(task_metrics.keys()):
            overall = task_metrics[baseline_name].get("overall", {})
            overall_rows.append(
                {
                    "task": task_name,
                    "baseline": baseline_name,
                    "precision": overall.get("precision", 0.0),
                    "recall": overall.get("recall", 0.0),
                    "f1": overall.get("f1", 0.0),
                }
            )

            per_source = task_metrics[baseline_name].get("per_source", {})
            for source in sorted(per_source.keys()):
                src_metrics = per_source[source]
                per_source_rows.append(
                    {
                        "task": task_name,
                        "source": source,
                        "baseline": baseline_name,
                        "precision": src_metrics.get("precision", 0.0),
                        "recall": src_metrics.get("recall", 0.0),
                        "f1": src_metrics.get("f1", 0.0),
                    }
                )

    overall_df = pd.DataFrame(overall_rows, columns=["task", "baseline", "precision", "recall", "f1"])
    per_source_df = pd.DataFrame(
        per_source_rows, columns=["task", "source", "baseline", "precision", "recall", "f1"]
    )
    output_dir = output_tsv_path
    if output_tsv_path.endswith(".tsv"):
        output_dir = os.path.dirname(output_tsv_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    overall_df.to_csv(os.path.join(output_dir, "overall_metrics.tsv"), sep="\t", index=False)
    per_source_df.to_csv(os.path.join(output_dir, "per_source_metrics.tsv"), sep="\t", index=False)


def write_prediction_labels_tsv(
    a_to_c_predictions: pd.DataFrame, b_to_c_predictions: pd.DataFrame, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base_cols = [
        "source",
        "doc_name",
        "paragraph_id",
        "sent_id_in_paragraph",
        "word",
        "true_label_for_word(0/1)",
    ]
    a_pred_cols = sorted([c for c in a_to_c_predictions.columns if c.endswith("_pred")])
    b_pred_cols = sorted([c for c in b_to_c_predictions.columns if c.endswith("_pred")])
    a_to_c_predictions[base_cols + a_pred_cols].to_csv(
        os.path.join(output_dir, "a_to_c_predictions.tsv"), sep="\t", index=False
    )
    b_to_c_predictions[base_cols + b_pred_cols].to_csv(
        os.path.join(output_dir, "b_to_c_predictions.tsv"), sep="\t", index=False
    )


def main() -> None:
    args = parse_args()
    punkt_train_documents = load_documents(path=args.data_path, split="train")
    if not punkt_train_documents:
        raise ValueError("No rows found for punkt training split='train' in input data.")
    documents = load_documents(path=args.data_path, split=args.split)
    if not documents:
        raise ValueError(f"No rows found for split={args.split!r} in {args.data_path}")

    baselines = build_baselines(train_documents=punkt_train_documents, ersatz_bin=args.ersatz_bin)
    if not baselines:
        raise RuntimeError("No baselines are available. Install dependencies or provide valid configuration.")

    a_to_c_metrics, a_to_c_predictions = evaluate_a_to_c(documents=documents, baselines=baselines)
    b_to_c_metrics, b_to_c_predictions = evaluate_b_to_c(documents=documents, baselines=baselines)

    report = {
        "split": args.split,
        "tasks": {
            "a_to_c": a_to_c_metrics,
            "b_to_c": b_to_c_metrics,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    if args.output_tsv_path:
        write_metrics_tsv(report=report, output_tsv_path=args.output_tsv_path)
    labels_output_dir = args.output_labels_dir
    if not labels_output_dir and args.output_tsv_path:
        labels_output_dir = args.output_tsv_path
    if labels_output_dir:
        write_prediction_labels_tsv(
            a_to_c_predictions=a_to_c_predictions,
            b_to_c_predictions=b_to_c_predictions,
            output_dir=labels_output_dir,
        )


if __name__ == "__main__":
    main()
