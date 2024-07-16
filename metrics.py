import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(preds, id2label):
    labels = preds.label_ids
    return get_metrics(preds.predictions, labels, id2label)

def get_metrics(preds, labels, id2label):
    ret = {}
    f1s = []
    precs = []
    recalls = []

    preds = preds.argmax(-1)

    for i, cat in id2label.items():

        cat_labels, cat_preds = labels == i, preds == i

        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels, cat_preds, average='binary', zero_division=0,
        )

        f1s.append(f1)
        precs.append(precision)
        recalls.append(recall)

        ret[cat.lower()+"_f1"] = f1
        ret[cat.lower()+"_precision"] = precision
        ret[cat.lower()+"_recall"] = recall

    _, _, micro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    ret["micro_f1"] = micro_f1
    ret["acc"] = accuracy_score(labels, preds)

    ret["macro_f1"] = torch.Tensor(f1s).mean()
    ret["macro_precision"] = torch.Tensor(precs).mean()
    ret["macro_recall"] = torch.Tensor(recalls).mean()

    return ret