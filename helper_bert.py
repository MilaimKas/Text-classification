# helper_bert.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -------------------------
# Simple result containers
# -------------------------

@dataclass
class TokenResults:
    tokenizer: Any
    train_enc: Dict[str, torch.Tensor]
    test_enc: Dict[str, torch.Tensor]
    train_dataset: Dataset
    test_dataset: Dataset


@dataclass
class BertTrained:
    model: Any
    trainer: Trainer


@dataclass
class PredictionResult:
    logits: torch.Tensor            # [N, 2]
    proba_pos: np.ndarray           # [N]
    pred_labels: np.ndarray         # [N]
    true_labels: np.ndarray         # [N]


# -------------------------
# Tokenization / datasets
# -------------------------

def create_token_datasets(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
) -> TokenResults:
    """
    Tokenize text and build HF Datasets with labels.

    Notes:
    - Uses padding+truncation (static) for simplicity.
    - Removes token_type_ids only if present.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(
        X_train.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        X_test.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_enc = dict(train_enc)
    test_enc = dict(test_enc)

    train_enc["labels"] = torch.tensor(y_train.astype(int).tolist(), dtype=torch.long)
    test_enc["labels"] = torch.tensor(y_test.astype(int).tolist(), dtype=torch.long)

    # Some models/tokenizers provide token_type_ids, others don't.
    train_enc.pop("token_type_ids", None)
    test_enc.pop("token_type_ids", None)

    # HF Dataset expects lists/arrays, not torch tensors; convert to python lists
    train_ds = Dataset.from_dict({k: v.cpu().numpy() for k, v in train_enc.items()})
    test_ds = Dataset.from_dict({k: v.cpu().numpy() for k, v in test_enc.items()})

    return TokenResults(
        tokenizer=tokenizer,
        train_enc=train_enc,
        test_enc=test_enc,
        train_dataset=train_ds,
        test_dataset=test_ds,
    )


# -------------------------
# Training
# -------------------------

def train_bert_classifier(
    train_dataset: Dataset,
    test_dataset: Dataset,
    model_name_or_path: str = "bert-base-uncased",
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    seed: int = 42,
    compute_metrics=None,
    output_attentions: bool = False,
    do_train: bool = True
) -> BertTrained:
    """
    Fine-tune a binary classifier.

    - model_name_or_path can be a checkpoint path or a base model.
    - compute_metrics is optional (keep lightweight).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=2,
        output_attentions=output_attentions,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        seed=seed,
        report_to="none",  # avoids wandb/tensorboard unless you want it
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    if do_train:
        trainer.train()
    
    return BertTrained(model=model, trainer=trainer)


# -------------------------
# Predictions
# -------------------------

@torch.no_grad()
def predict_with_model(
    model,
    tokenizer,
    texts: Union[List[str], np.ndarray],
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
):
    """
    Predict logits, probabilities, and labels using ONLY model + tokenizer.
    No Trainer required.

    Returns:
        dict with keys:
          - logits: np.ndarray [N, 2]
          - proba_pos: np.ndarray [N]
          - pred_labels: np.ndarray [N]
    """
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        all_logits.append(out.logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    return {
        "logits": logits,
        "proba_pos": probs[:, 1],
        "pred_labels": probs.argmax(axis=-1),
    }


def get_predictions_trainer(
    trained: BertTrained,
    token_data: TokenResults,
) -> PredictionResult:
    """
    Predict on the test dataset and return logits, P(y=1), labels, true labels.
    """
    pred_out = trained.trainer.predict(token_data.test_dataset)

    logits = torch.tensor(pred_out.predictions)  # [N, 2]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    proba_pos = probs[:, 1]
    pred_labels = probs.argmax(axis=-1)

    # True labels from dataset
    true_labels = np.array(token_data.test_dataset["labels"], dtype=int)

    return PredictionResult(
        logits=logits,
        proba_pos=proba_pos,
        pred_labels=pred_labels,
        true_labels=true_labels,
    )

def get_predictions(
    *,
    trained: BertTrained | None = None,
    model=None,
    tokenizer=None,
    texts: List[str] | None = None,
    token_data: TokenResults | None = None,
    batch_size: int = 32,
    max_length: int = 256,
):
    """
    Unified prediction interface.

    Usage patterns:
    1) Trainer-based:
       get_predictions(trained=trained_obj, token_data=token_data)

    2) Model-only:
       get_predictions(model=model, tokenizer=tokenizer, texts=X)
    """

    # --- Case 1: Trainer-based
    if trained is not None:
        if token_data is None:
            raise ValueError("token_data must be provided when using trained (Trainer-based).")
        return get_predictions_trainer(trained, token_data)

    # --- Case 2: Model-only
    if model is not None and tokenizer is not None and texts is not None:
        return predict_with_model(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
        )

    raise ValueError(
        "Invalid call. Provide either:\n"
        "  - trained + token_data (Trainer-based), or\n"
        "  - model + tokenizer + texts (model-only)"
    )



# -------------------------
# Integrated Gradients (positive class)
# -------------------------

@torch.no_grad()
def _predict_proba_pos(model, **encoded) -> float:
    out = model(**encoded)
    probs = torch.softmax(out.logits, dim=-1)
    return float(probs[0, 1].item())


def integrated_gradients_positive(
    model,
    tokenizer,
    text: str,
    steps: int = 64,
    baseline: str = "pad",  # "pad" | "mask" | "unk"
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Integrated Gradients for the POSITIVE class logit (index 1).

    Returns dict with tokens and per-token attributions.
    """
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    input_ids = enc["input_ids"]  # [1, T]
    attention_mask = enc.get("attention_mask", None)
    token_type_ids = enc.get("token_type_ids", None)

    if baseline == "pad":
        base_id = tokenizer.pad_token_id or tokenizer.unk_token_id
    elif baseline == "mask":
        base_id = tokenizer.mask_token_id or tokenizer.unk_token_id
    elif baseline == "unk":
        base_id = tokenizer.unk_token_id
    else:
        raise ValueError("baseline must be one of: 'pad', 'mask', 'unk'")

    baseline_ids = torch.full_like(input_ids, fill_value=base_id)

    # Keep CLS/SEP unchanged
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is not None:
        baseline_ids[input_ids == cls_id] = cls_id
    if sep_id is not None:
        baseline_ids[input_ids == sep_id] = sep_id

    embed_layer = model.get_input_embeddings()
    input_emb = embed_layer(input_ids)
    base_emb = embed_layer(baseline_ids)
    delta = input_emb - base_emb

    total_grad = torch.zeros_like(input_emb)

    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        emb = base_emb + alpha * delta
        emb.requires_grad_(True)

        out = model(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pos_logit = out.logits[0, 1]
        grad = torch.autograd.grad(pos_logit, emb, retain_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / float(steps)
    ig = (delta * avg_grad).sum(dim=-1).squeeze(0)  # [T]

    if attention_mask is not None:
        ig = ig * attention_mask.squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    attributions = ig.detach().cpu().tolist()

    max_abs = max(1e-12, max(abs(x) for x in attributions))
    attributions_norm = [x / max_abs for x in attributions]

    with torch.no_grad():
        p_pos = _predict_proba_pos(model, **enc)

    return {
        "text": text,
        "p_pos": p_pos,
        "tokens": tokens,
        "attributions": attributions,
        "attributions_norm": attributions_norm,
    }


# -------------------------
# Visualization
# -------------------------

def plot_ig(
    model,
    tokenizer,
    text: str,
    steps: int = 64,
    baseline: str = "pad",
    thresh: float = 0.1,
    max_length: Optional[int] = None,
):
    """
    Simple bar plot of normalized IG attributions above |thresh|.
    """
    result = integrated_gradients_positive(
        model,
        tokenizer,
        text=text,
        steps=steps,
        baseline=baseline,
        max_length=max_length,
    )

    rows = []
    for tok, score in zip(result["tokens"], result["attributions_norm"]):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if abs(score) >= thresh:
            rows.append((tok, score))

    if not rows:
        print(f"No tokens above threshold |score| >= {thresh}. P(y=1)={result['p_pos']:.3f}")
        return

    df = pd.DataFrame(rows, columns=["token", "norm_score"])

    plt.figure(figsize=(max(6, len(df) * 0.4), 4))
    plt.bar(df["token"], df["norm_score"])
    plt.xticks(rotation=90)
    plt.title(f"P(y=1) = {result['p_pos']:.3f}")
    plt.tight_layout()
    plt.show()

    return df
