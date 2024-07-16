from datasets import load_dataset, VerificationMode
from training import train_and_eval, load_model
from tuning import hyperparameter_sweep, get_training_arguments
from preprocess import preprocess_text

task_name = "emotion"

def load_datasets(preprocess=True):

    ds = load_dataset(
        f"hessan/emotion",
        verification_mode=VerificationMode.NO_CHECKS,
    )

    if preprocess:
        ds = ds.map(preprocess_text)

    return ds


def train(base_model, use_defaults_if_not_tuned=False, **kwargs):

    ds = load_datasets()

    id2label = {k: v for k, v in enumerate(ds["train"].features["label"].names)}

    training_args = get_training_arguments(
        base_model,
        task_name=task_name,
        metric_for_best_model="eval/macro_f1",
        use_defaults_if_not_tuned=use_defaults_if_not_tuned,
    )

    return train_and_eval(
        base_model=base_model,
        dataset=ds,
        id2label=id2label,
        training_args=training_args,
        **kwargs,
    )


def hp_tune(base_model, **kwargs):

    ds = load_datasets()

    id2label = {k: v for k, v in enumerate(ds["train"].features["label"].names)}

    def model_init():
        model, _ = load_model(base_model, id2label)
        return model

    _, tokenizer = load_model(base_model, id2label)

    config_info = {
        "model": base_model,
        "task": task_name
    }

    return hyperparameter_sweep(
        name=f"swp-{task_name}-{base_model}",
        group_name=f"swp-{task_name}",
        model_init=model_init,
        tokenizer=tokenizer,
        datasets=ds,
        id2label=id2label,
        config_info=config_info,
        **kwargs,
    )
