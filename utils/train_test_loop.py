import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import os
import pickle
import numpy as np
import random
import time


seed = 42  # You can choose any integer seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

'''# If using CUDA, set the seed for GPU as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
'''

def train_and_evaluate(model, trainloader, testloader, optimizer, loss_fn, num_epochs, corr_p, rep = None, sound=10, loss_type = None, clothing = False):
    seed = 42  # You can choose any integer seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    initial_lr = optimizer.param_groups[0]['lr']

    # Initialize a list to store epoch data
    results = []

    #Only when debbuging
    #torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct_train = 0

        for inputs, vl, targets in trainloader:
            #if loss_type == 'Supervised':
            #    train_targets = torch.max(targets, dim=1)[1]
            inputs, vl, targets = inputs.to(device), vl.to(device), targets.to(device)
            

            optimizer.zero_grad()
            outputs = model(inputs)
            #if loss_type == 'Supervised':
            #    # For cross-entropy loss, targets should be class indices
            #    loss = loss_fn(outputs, train_targets)
            #else:
            loss = loss_fn(outputs, vl) 
            loss.backward()
            optimizer.step()

            # Update batch's loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct_train += torch.sum(preds == true)

        train_acc = correct_train.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        # Evaluate the model on the test set
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for inputs, targets in testloader: 
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, preds = torch.max(outputs, dim=1)
                _, true = torch.max(targets, dim=1)
                correct_test += torch.sum(preds == true)

        test_acc = correct_test.double() / len(testloader.dataset)

        # Calculate detached loss 
        detached_train_loss = 0.0
        detached_test_loss = 0.0
        with torch.no_grad():
            det_loss_fn = torch.nn.CrossEntropyLoss()  
            for inputs, _, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                detached_train_loss += det_loss_fn(outputs, targets).item()
            detached_train_loss /= len(trainloader.dataset)

            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                detached_test_loss += det_loss_fn(outputs, targets).item()
            detached_test_loss /= len(testloader.dataset)

        # Get the actual learning rate from the optimizer
        actual_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        # Store results for this epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'test_acc': test_acc.item(),
            'train_detached_loss': detached_train_loss,
            'test_detached_loss': detached_test_loss,
            'optimizer': type(optimizer).__name__,
            'loss_fn': loss_type,
            'repetition': rep,
            'initial_lr': initial_lr,
            'actual_lr': actual_lr,
            'corr_p': corr_p,
            'epoch_time': epoch_time,
        }
        results.append(epoch_data)

        if epoch % sound == sound - 1:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
                  f'Train Detached Loss: {detached_train_loss:.4f}, Test Detached Loss: {detached_test_loss:.4f}, '
                  f'Learning Rate: {actual_lr:.6f}, Epoch Time: {epoch_time:.2f} seconds')

    # Convert results to DataFrame at the end
    results_df = pd.DataFrame(results)

    return model, results_df