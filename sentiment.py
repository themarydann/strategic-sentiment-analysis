import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text
from training import train_and_eval, load_model
from tuning import hyperparameter_sweep, get_training_arguments

task_name = "sentiment"

id2label = {0: 'NEG', 1: 'NEU', 2: 'POS'}
label2id = {v: k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, "datasets")

def load_df(path):

    df = pd.read_csv(path, encoding="latin-1")

    matches = {"NEG": "negative", "NEU": "neutral", "POS": "positive"}

    for label, idx in label2id.items():
        replacement = matches[label]
        df.loc[df["label"] == replacement, "label"] = idx

    df["label"] = df["label"].astype(int)
    return df

def load_datasets(seed=2021, preprocess=True, **kwargs):

    train_df = load_df(
        os.path.join(data_dir, f"{task_name}_train.csv")
    )

    test_df = load_df(
        os.path.join(data_dir, f"{task_name}_test.csv")
    )

    train_df, dev_df = train_test_split(train_df, test_size=0.2)

    if preprocess:
        train_df["text"] = train_df["text"].apply(preprocess_text)
        dev_df["text"] = dev_df["text"].apply(preprocess_text)
        test_df["text"] = test_df["text"].apply(preprocess_text)

    features = Features({
        'id': Value('int64'),
        'text': Value('string'),
        'label': ClassLabel(num_classes=3, names=["NEG", "NEU", "POS"])
    })

    columns = ["text", "id", "label"]

    train_dataset = Dataset.from_pandas(
        train_df[columns],
        features=features,
        preserve_index=False
    )
    dev_dataset = Dataset.from_pandas(
        dev_df[columns],
        features=features,
        preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        test_df[columns],
        features=features,
        preserve_index=False
    )

    return DatasetDict(
        train=train_dataset,
        dev=dev_dataset,
        test=test_dataset
    )

def train(
    base_model, use_defaults_if_not_tuned=False, **kwargs
):

    ds = load_datasets()

    training_args = get_training_arguments(
        base_model, task_name=task_name,
        metric_for_best_model="eval/macro_f1", use_defaults_if_not_tuned=use_defaults_if_not_tuned
    )

    return train_and_eval(
        base_model=base_model, dataset=ds, id2label=id2label,
        training_args=training_args, **kwargs
    )


def hp_tune(model_name, **kwargs):

    task_name = "sentiment"

    ds = load_datasets()

    def model_init():
        model, _ = load_model(model_name, id2label)
        return model

    _, tokenizer = load_model(model_name, id2label)

    config_info = {
        "model": model_name,
        "task": task_name
    }

    return hyperparameter_sweep(
        name=f"swp-{task_name}-{model_name}",
        group_name=f"swp-{task_name}",
        model_init=model_init,
        tokenizer=tokenizer,
        datasets=ds,
        id2label=id2label,
        config_info=config_info,
        **kwargs,
    )