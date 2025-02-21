import wandb
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from metrics import compute_metrics as _compute_metrics

# hyperparameters
parameters_dict = {
    "epochs": {
        "values": [3, 4, 5],
    },
    "batch_size": {"value": 32},
    "learning_rate": {"values": [2e-5, 3e-5, 5e-5, 6e-5, 7e-5, 8e-5, 1e-4]},
    "weight_decay": {"value": 0.1},
    "warmup_ratio": {"values": [0.06, 0.08, 0.10]},
}

def get_training_arguments(
    model_name,
    task_name,
    metric_for_best_model,
    use_defaults_if_not_tuned=False,
    dont_report=False,
):
    
    try:
        api = wandb.Api()
        sweeps = api.project("hessan").sweeps()
        # Get project sweep
        sweep = next(
            sweep
            for sweep in sweeps
            if sweep.name == f"swp-{task_name}-{model_name}"
        )

        # Get best run
        best_run = sweep.best_run(metric_for_best_model)

        # Get best run config
        tuned_params = best_run.config

    except StopIteration:
        if use_defaults_if_not_tuned:
            tuned_params = {}
        else:
            error_msg = f"Model {model_name} not tuned for task {task_name}. Use use_defaults_if_not_tuned=True to use default training arguments."
            raise ValueError(error_msg)

    args = TrainingArguments(
        output_dir=f"./results/{task_name}-{model_name}",
        num_train_epochs=tuned_params.get("epochs", 3),
        per_device_train_batch_size=tuned_params.get("batch_size", 32),
        per_device_eval_batch_size=tuned_params.get("batch_size", 32),
        gradient_accumulation_steps=tuned_params.get("accumulation_steps", 1),
        warmup_ratio=tuned_params.get("warmup_ratio", 0.1),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=tuned_params.get("learning_rate", 5e-5),
        do_eval=False,
        weight_decay=tuned_params.get("weight_decay", 0.01),
        fp16=False,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        group_by_length=True,
        overwrite_output_dir=True,
    )

    if dont_report:
        args.report_to = None

    return args


def hyperparameter_sweep(
    name,
    group_name,
    model_init,
    tokenizer,
    datasets,
    id2label,
    sweep_method="random",
    format_dataset=None,
    compute_metrics=None,
    config_info=None,
    tokenize_fun=None,
    metric_for_best_model="eval_macro_f1",
    count=None,
):
    
    sweep_config = {
        "name": name,
        "method": sweep_method,
        "parameters": parameters_dict,
    }

    if compute_metrics is None:

        def compute_metrics(preds):
            return _compute_metrics(preds, id2label)

    def _tokenize_fun(x):
        if tokenize_fun:
            return tokenize_fun(x)
        else:
            return tokenizer(x["text"], padding=False, truncation=True)

    tokenized_ds = datasets.map(_tokenize_fun, batched=True)

    if format_dataset is not None:
        tokenized_ds = tokenized_ds.map(format_dataset)

    tokenized_ds = tokenized_ds.remove_columns(
        [x for x in datasets["train"].column_names if x not in ["labels", "label"]]
    )

    def train(config=None):
        init_params = {
            "config": config or {},
            "group": group_name,
            "job_type": "sweep",
        }

        if config_info:
            init_params["config"].update(config_info)

        with wandb.init(**init_params):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            training_args = TrainingArguments(
                output_dir="./tmp/sweeps",
                report_to="wandb",  # Turn on Weights & Biases logging
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.batch_size,
                warmup_ratio=config.warmup_ratio,
                per_device_eval_batch_size=config.batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                remove_unused_columns=False,
                group_by_length=True,
                fp16=False,
            )

            # define training loop
            trainer = Trainer(
                # model,
                model_init=model_init,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["dev"],
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
            )

            # start training loop
            trainer.train()

    # Initiate sweep
    sweep_id = wandb.sweep(sweep_config, project='HESSAN')

    wandb.agent(sweep_id, train, count=count)
