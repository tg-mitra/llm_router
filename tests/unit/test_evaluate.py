from evaluate import evaluate_predictions


class TestEvaluatePredictions:
    def test_perfect_predictions(self):
        y_true = ["simple", "coding", "simple", "coding"]
        y_pred = ["simple", "coding", "simple", "coding"]
        report = evaluate_predictions(y_true, y_pred, labels=["coding", "simple"])
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0
        assert report.per_class["simple"]["precision"] == 1.0
        assert report.per_class["coding"]["recall"] == 1.0

    def test_confusion_matrix_shape(self):
        labels = ["coding", "simple"]
        report = evaluate_predictions(["simple", "coding"], ["coding", "coding"], labels=labels)
        assert len(report.confusion_matrix) == 2
        assert len(report.confusion_matrix[0]) == 2

    def test_to_dict_roundtrip_keys(self):
        report = evaluate_predictions(["simple"], ["simple"], labels=["simple"])
        payload = report.to_dict()
        assert set(payload.keys()) == {"accuracy", "macro_f1", "labels", "per_class", "confusion_matrix"}

    def test_imperfect_predictions_reduce_accuracy(self):
        y_true = ["simple", "simple", "coding", "coding"]
        y_pred = ["simple", "coding", "coding", "coding"]
        report = evaluate_predictions(y_true, y_pred, labels=["coding", "simple"])
        assert report.accuracy == 0.75
        assert report.per_class["simple"]["recall"] == 0.5
