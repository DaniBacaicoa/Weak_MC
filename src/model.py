import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.0, bn = False, activation='relu'):
        super().__init__()

        # Create a list of layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create a list of linear layers using ModuleList
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
        for layer in self.layers:
            #nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.weight, 0.1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.01)
                

        # Create a list of batch normalization layers using ModuleList
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i+1])
            for i in range(len(hidden_sizes))
        ])

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout_p)
        self.bn = bn
        self.activation = activation

    def forward(self, x):
        # Iterate over the linear layers and apply them sequentially to the input
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)
        # Apply the final linear layer to get the output
        x = self.layers[-1](x)
        return x
    

class ResNet_18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_18, self).__init__()
        

        self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
    
        self.resnet.maxpool = nn.Identity()
        
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.resnet(x)
    

class ResNet50(nn.Module):
    """
    A ResNet-50 model pre-trained on ImageNet, adapted for the Clothing1M dataset.

    The final fully connected layer is replaced to match the number of classes
    in the Clothing1M dataset (14 classes).
    """
    def __init__(self, num_classes=14, fine_tune_all=False):

        super(ResNet50, self).__init__()

        self.fine_tune_all = fine_tune_all

        weights = models.ResNet50_Weights.DEFAULT

        self.resnet50 = models.resnet50(weights=weights)
        num_ftrs = self.resnet50.fc.in_features

        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        if not fine_tune_all:
            print("Freezing base model parameters. Only the final classifier will be trained.")
            # Freeze all parameters first
            for param in self.resnet50.parameters():
                param.requires_grad = False
            # Unfreeze the parameters of the final layer (fc)
            for param in self.resnet50.fc.parameters():
                param.requires_grad = True
        else:
            print("All model parameters will be fine-tuned.")


    def forward(self, x):

        return self.resnet50(x)