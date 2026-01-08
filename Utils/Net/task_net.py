import torch.nn as nn
import torch.nn.functional as F




class DeepLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[128, 64], output_size=1, dropout=0.3):
        super(DeepLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_sizes[1], 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.norm1(out)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        last_time_step = out[:, -1, :]
        out = F.gelu(self.fc1(last_time_step))
        out = self.fc2(out)

        return out
