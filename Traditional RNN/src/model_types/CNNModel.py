import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.original_cnn_model = torch.load("/home/user/Documents/GitHub/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth")

        # Freeze all parameters in the original model
        for param in self.original_cnn_model.parameters():
            param.requires_grad = False

        # Remove the final classification layers (usually the last fully connected layer)
        self.features = nn.Sequential(*list(self.original_cnn_model.children())[:-1])
        self.logits = self.original_cnn_model.fc

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        y = self.logits(x)
        return x, y
    
    def get_feature_dim(self):
        return self.original_cnn_model.fc.in_features
    
    def get_num_classees(self):
        return self.original_cnn_model.fc.out_features