import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

import torch.nn as nn
import torch.optim as optim

def evaluate(model, x_test, y_test, device):
    """
    Evaluates the model on the test set using true labels.

    Args:
        model (nn.Module): The model to evaluate.
        x_test (torch.Tensor): Test data inputs.
        y_test (torch.Tensor): Test data true labels (in one hot encoding).
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval()
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    with torch.no_grad():
        outputs = model(x_test)
        if outputs.shape[1] > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()

    preds_cpu = preds.cpu().numpy()
    y_test_cpu = y_test.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_cpu, preds_cpu, average='binary' if len(np.unique(y_test_cpu)) == 2 else 'macro', zero_division=0
    )
    accuracy = accuracy_score(y_test_cpu, preds_cpu)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def train_and_evaluate(
    model,
    train_data,
    test_data,
    loss_fn,
    optimizer,
    epochs,
    device,
    print_every=10
):
    """
    Trains a model using weak labels and evaluates it using true labels.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_data (tuple): A triplet (x_train, w_train, y_train) where
                            x_train are the inputs, w_train are the weak
                            labels for the loss, and y_train are the true
                            labels for periodic evaluation.
        test_data (tuple): A tuple (x_test, y_test) for final evaluation.
        loss_fn: The loss function.
        optimizer: The optimizer.
        epochs (int): The number of training epochs.
        device (torch.device): The device (e.g., 'cuda' or 'cpu').
        print_every (int): The interval of epochs to print metrics.

    Returns:
        dict: A dictionary containing the history of evaluation metrics.
    """
    x_train, w_train, y_train = train_data
    x_test, y_test = test_data

    x_train, w_train = x_train.to(device), w_train.to(device)
    
    history = {'train_metrics': {}, 'test_metrics': {}}

    for epoch in range(1, epochs + 1):
        model.train()
        
        # Forward pass
        outputs = model(x_train)
        loss = loss_fn(outputs, w_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == epochs:
            # Evaluate on both training (with true labels) and test sets
            train_metrics = evaluate(model, x_train, y_train, device)
            test_metrics = evaluate(model, x_test, y_test, device)
            
            history['train_metrics'][epoch] = train_metrics
            history['test_metrics'][epoch] = test_metrics
            
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Train Metrics -> "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1_score']:.4f}")
            print(f"  Test Metrics  -> "
                  f"Acc: {test_metrics['accuracy']:.4f}, "
                  f"F1: {test_metrics['f1_score']:.4f}")
            print("-" * 30)

    return history

if __name__ == '__main__':
    # Example Usage
    # 1. Define parameters
    INPUT_SIZE = 10
    NUM_CLASSES = 2  # Binary classification
    NUM_TRAIN_SAMPLES = 1000
    NUM_TEST_SAMPLES = 200
    EPOCHS = 100
    LEARNING_RATE = 0.01
    PRINT_EVERY = 20

    # 2. Check for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Create a simple model
    # For binary classification, output size is 1 for BCEWithLogitsLoss
    output_size = 1 if NUM_CLASSES == 2 else NUM_CLASSES
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 32),
        nn.ReLU(),
        nn.Linear(32, output_size)
    ).to(device)

    # 4. Create dummy data
    # Training data
    x_train = torch.randn(NUM_TRAIN_SAMPLES, INPUT_SIZE)
    y_train_true = torch.randint(0, NUM_CLASSES, (NUM_TRAIN_SAMPLES,)).float()
    
    # Simulate weak labels (e.g., true labels with 20% noise)
    noise = (torch.rand(NUM_TRAIN_SAMPLES) < 0.2).long()
    w_train_weak = torch.abs(y_train_true - noise).float() # Flip some labels
    
    # Reshape labels for loss function if needed
    w_train_weak = w_train_weak.view(-1, 1)
    
    train_data = (x_train, w_train_weak, y_train_true)

    # Test data
    x_test = torch.randn(NUM_TEST_SAMPLES, INPUT_SIZE)
    y_test_true = torch.randint(0, NUM_CLASSES, (NUM_TEST_SAMPLES,)).float()
    test_data = (x_test, y_test_true)

    # 5. Define loss and optimizer
    # Use BCEWithLogitsLoss for binary classification, it's more stable
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Run training and evaluation
    metrics_history = train_and_evaluate(
        model=model,
        train_data=train_data,
        test_data=test_data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        print_every=PRINT_EVERY
    )

    print("\nTraining finished.")
    print("Final Test Metrics:", metrics_history['test_metrics'][EPOCHS])