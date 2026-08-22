from train import MODEL_PATH, train


def test_train_saves_model_and_meets_accuracy():
    accuracy = train()

    assert MODEL_PATH.exists()
    assert accuracy >= 0.9