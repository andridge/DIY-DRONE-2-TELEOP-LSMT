#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import re

class TeleopLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, output_size=2):
        super(TeleopLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def next_version_number(model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    versions = []

    for f in os.listdir(model_dir):
        match = re.match(r"lstm_v(\d+)\.pth", f)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1


def create_versioned_lstm(
    input_size=10, hidden_size=64, num_layers=1, output_size=2
):
    version = next_version_number()
    filename = f"lstm_v{version}.pth"
    filepath = os.path.join("models", filename)

    print(f"➡️ Creating LSTM model version: v{version}")

    model = TeleopLSTM(input_size, hidden_size, num_layers, output_size)
    torch.save(model.state_dict(), filepath)

    print(f"✅ Saved new model: {filepath}")


if __name__ == "__main__":
    create_versioned_lstm()

