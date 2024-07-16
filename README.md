
# Prerequisites 

* `Python 3.6` or higher
* Install necessary libraries using `pip install -r requirements.txt`

# Training models

To train a model, use the script `train.py`, which takes the following arguments:


```bash
python train.py --base_model <model_name> \
    --task <task> \
```

For instance, to train a RoBERTa model for Sentiment Analysis, use:

```bash
python bin/train.py --base_model "google-bert/bert-base-uncased"\
    --task sentiment
```

## Benchmarking

To run benchmarks you can use also `train.py` passing the `--benchmark`

```bash

python bin/train.py --base_model "google-bert/bert-base-uncased"\
    --task sentiment --benchmark --times 10
```