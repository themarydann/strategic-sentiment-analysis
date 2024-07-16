import sentiment, emotion, topic, fire, wandb
from transformers.trainer_utils import set_seed
import time

modules = {
    "sentiment": sentiment,
    "emotion": emotion,
    "topic": topic
}

def get_wandb_run_info(base_model, task):
    # Check if task module has a get_wandb_run_info method

    return {
        "project": 'HESSAN',
        # Group by model name
        "group": f"{task}",
        "job_type": f"{task}-{base_model.split('/')[-1]}",
        # Name run by model name
        "config": {
            "model": base_model,
            "task": task,
        }
    }

def train(
    base_model, task=None,
    output_path=None,
    limit=None, 
    benchmark=False, times=10,
    **kwargs
):
    #print(kwargs)
    train_args = kwargs.copy()
    if limit:
        train_args["limit"] = limit

    if not benchmark:
        set_seed(int(time.time()))
        task_run = modules[task].train

        print(
            f"Training {base_model} for {task}"
        )

        trainer, test_results = task_run(
            base_model, dont_report=True,
            **train_args
        )

        print(f"Saving model to {output_path}")
        trainer.save_model(output_path)

    else:
        for i in range(times):
            print(f"{i+1} Iteration")
            # if wandb configured

            set_seed(int(time.time()))
            print(
                f"Training {base_model} for {task}"
            )

            wandb_run = None
            try:
                wandb_run_info = get_wandb_run_info(
                    base_model, task, **kwargs
                )
                wandb_run = wandb.init(
                    reinit=True,
                    **wandb_run_info
                )

                train_args["report_to"] = "wandb"
            except KeyError as e:
                print(f"WANDB not configured. Skipping: {e}")

            task_fun = modules[task].train
            trainer, test_results = task_fun(
                base_model,
                **train_args
            )

            metrics = test_results.metrics

            if wandb_run:
                for k, v in metrics.items():
                    wandb.log({k: v})

            wandb_run.finish()


if __name__ == "__main__":
    fire.Fire(train)