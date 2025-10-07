from sklearn.metrics import accuracy_score

def get_sum_metrics(predictions, metrics=[]):
    for i in range(3):
        metrics.append(lambda x: x + i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics

pred = 1
metrics = [accuracy_score()]

print(get_sum_metrics(pred, metrics=metrics))