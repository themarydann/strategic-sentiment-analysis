import torch
import os
import torch
import tempfile
from metrics import compute_metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

def load_model(
    base_model, id2label, max_length=128, auto_class=AutoModelForSequenceClassification,
    problem_type=None,
):

    print(f"Loading model {base_model}")
    model = auto_class.from_pretrained(
        base_model, return_dict=True, num_labels=len(id2label),
        problem_type=problem_type
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = max_length

    if type(id2label) is not dict:
        id2label = {str(i): label for i, label in enumerate(id2label)}
    label2id = {label: i for i, label in id2label.items()}

    model.config.id2label = id2label
    model.config.label2id = label2id

    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def train_huggingface(
        base_model, dataset, id2label,
        metrics_fun, training_args,
        max_length=128, auto_class=AutoModelForSequenceClassification,
        format_dataset=None, use_dynamic_padding=True, class_weight=None, trainer_class=None,  data_collator_class=DataCollatorWithPadding, tokenize_fun=None, problem_type=None,
        **kwargs
    ):


    padding = False if use_dynamic_padding else 'max_length'

    model, tokenizer = load_model(
        base_model, id2label=id2label,
        max_length=max_length, auto_class=auto_class,
        problem_type=problem_type
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    def _tokenize_fun(x):
        if tokenize_fun:
            return tokenize_fun(x)
        else:
            return tokenizer(x['text'], padding=padding, truncation=True)

    dataset = dataset.map(
        _tokenize_fun, batched=True
    )

    if use_dynamic_padding:
        data_collator = data_collator_class(tokenizer, padding="longest")
    else:
        if not format_dataset:
            raise ValueError("Must provide format_dataset if not using dynamic padding")

    if format_dataset:
        for split in dataset.keys():
            dataset[split] = format_dataset(dataset[split])

    output_path = tempfile.mkdtemp(
        prefix="hessan",
    )

    trainer_args = {
        "model": model,
        "args": training_args,
        "compute_metrics": metrics_fun,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["dev"],
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }

    trainer_class = trainer_class or Trainer

    trainer = trainer_class(**trainer_args)

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    os.system(f"rm -Rf {output_path}")

    return trainer, test_results


def train_and_eval(base_model, dataset, id2label, limit=None, metrics_fun=None, **kwargs):

    if limit:
        # smoke test
        dataset = dataset.select(range(limit))

    if type(id2label) is list:
        id2label = {i: label for i, label in enumerate(id2label)}

    label2id = {v: k for k, v in id2label.items()}

    if not metrics_fun:
        def metrics_fun(x): return compute_metrics(x, id2label=id2label)

    return train_huggingface(
        base_model=base_model, dataset=dataset, id2label=id2label, metrics_fun=metrics_fun, **kwargs
    )