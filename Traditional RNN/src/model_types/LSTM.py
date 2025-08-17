import torch
import torch.nn as nn

import torch.nn.init as init

class TraditionalBidirectionalLSTM(nn.Module):

    def __init__(self, feature_dim=6153, num_classes=3, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        feature_dim: Dimension of CNN feature output (2048 for ResNet-50's second-to-last layer)
        num_classes: Number of classes predicted by the CNN (3 for your 3 classes: 0, 1, 2)
        """
        super().__init__()
        
        # Update input_dim to the combined size (features + predictions)
        # input_dim = feature_dim + num_classes  # Combine CNN features and predictions
        
        # LSTM setup for sequence input (CNN features and predictions combined)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layer to output transition score
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

        self.apply_xavier_init()

    def apply_xavier_init(self):
        # Apply Xavier initialization to LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)  # or init.xavier_normal_(param)
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zero

                hidden_size = param.shape[0] // 4
                param.data[hidden_size:2*hidden_size] = 1.0
        
        # Apply Xavier initialization to fully connected layer
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: Input tensor of combined CNN features and predictions (shape: [B, S, input_dim])
        """
        # Forward pass through the LSTM layer
        features, _ = self.lstm(x)  # (B, seq_len, 2*hidden_dim)
        
        # Apply dropout to the last time step
        out = self.dropout(features[:, -1, :])  # Last timestep
        
        # Output from fully connected layer
        out = self.fc(out)
        
        # Sigmoid output for binary classification (transition/no-transition)
        # Probability of Positive (1) Class [Transition in this case]
        return features, out.squeeze(1), torch.sigmoid(out).squeeze(1)  # Output shape: (B,)