from sklearn.metrics import accuracy_score, classification_report, f1_score
from seqeval.metrics import classification_report as entity_report
from seqeval.metrics import f1_score as entity_f1
from seqeval.scheme import IOB2


def classification_metrics(labels, predictions, vocab):
    classes = list(range(vocab.n_labels))
    names = [str(vocab.i2l[index]) for index in classes]
    return {"accuracy": float(accuracy_score(labels, predictions)), "f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)), "report": classification_report(labels, predictions, labels=classes, target_names=names, digits=4, zero_division=0)}


def ner_metrics(labels, predictions):
    score = entity_f1(labels, predictions, mode="strict", scheme=IOB2, zero_division=0)
    report = entity_report(labels, predictions, mode="strict", scheme=IOB2, digits=4, zero_division=0)
    return {"f1": float(score), "report": report}
