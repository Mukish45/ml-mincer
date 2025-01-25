def evaluate_accuracy(model, validation_dataset):
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in validation_dataset:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy