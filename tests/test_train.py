from train import MODEL_PATH, train
import joblib


def test_train_saves_model_and_meets_accuracy():
    accuracy = train()

    assert MODEL_PATH.exists()
    assert accuracy >= 0.85

    loaded_model = joblib.load(MODEL_PATH)
    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_model, "classes_")
