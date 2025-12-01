#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import re
import rospy
from std_msgs.msg import String
import json
import threading
import websocket
import numpy as np


# -------------------------------------------
# LSTM MODEL (GPU Enabled)
# -------------------------------------------
class TeleopLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, output_size=2):
        """ IMU: ax, ay, az, gx, gy, gz  (6 values) → cmd_vel: linear_x, angular_z """
        super(TeleopLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)
        
        print(f"Model initialized on: {self.device}")

    def forward(self, x):
        # Move input to same device as model
        x = x.to(self.device)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# -------------------------------------------
# DATASET FOR IMU → CMD_VEL (Live from Tunnel)
# -------------------------------------------
class LiveIMUDataset(Dataset):
    def __init__(self, seq_len=20, max_samples=1000):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.imu_buffer = []  # Store IMU data: [ax, ay, az, gx, gy, gz]
        self.cmd_buffer = []  # Store command data: [linear_x, angular_z]
        self.lock = threading.Lock()
        
        # ROS subscriber for real data from tunnel
        rospy.init_node('live_training_dataset', anonymous=True)
        self.imu_sub = rospy.Subscriber('/august/imu/data', String, self.imu_callback)
        self.cmd_sub = rospy.Subscriber('/cmd_vel_ai', String, self.cmd_callback)
        
        print("Live dataset initialized - waiting for tunnel data...")

    def imu_callback(self, msg):
        """Receive IMU data through tunnel"""
        try:
            data = json.loads(msg.data)
            imu_values = [
                data.get('ax', 0.0), data.get('ay', 0.0), data.get('az', 0.0),
                data.get('gx', 0.0), data.get('gy', 0.0), data.get('gz', 0.0)
            ]
            
            with self.lock:
                self.imu_buffer.append(imu_values)
                # Keep buffer size manageable
                if len(self.imu_buffer) > self.max_samples:
                    self.imu_buffer.pop(0)
                    
        except Exception as e:
            print(f"IMU callback error: {e}")

    def cmd_callback(self, msg):
        """Receive command data through tunnel"""
        try:
            data = json.loads(msg.data)
            cmd_values = [
                data.get('linear_x', 0.0),
                data.get('angular_z', 0.0)
            ]
            
            with self.lock:
                self.cmd_buffer.append(cmd_values)
                # Keep buffer size manageable
                if len(self.cmd_buffer) > self.max_samples:
                    self.cmd_buffer.pop(0)
                    
        except Exception as e:
            print(f"CMD callback error: {e}")

    def __len__(self):
        with self.lock:
            return max(0, min(len(self.imu_buffer), len(self.cmd_buffer)) - self.seq_len)

    def __getitem__(self, idx):
        with self.lock:
            # Ensure we have enough data for sequence
            if idx + self.seq_len >= len(self.imu_buffer) or idx + self.seq_len >= len(self.cmd_buffer):
                # Return zeros if not enough data (shouldn't happen with proper bounds checking)
                seq = torch.zeros(self.seq_len, 6, dtype=torch.float32)
                tgt = torch.zeros(2, dtype=torch.float32)
                return seq, tgt
            
            seq_data = self.imu_buffer[idx:idx + self.seq_len]
            target_data = self.cmd_buffer[idx + self.seq_len]
            
            seq = torch.tensor(seq_data, dtype=torch.float32)
            tgt = torch.tensor(target_data, dtype=torch.float32)
            
            return seq, tgt


# -------------------------------------------
# CSV DATASET (Fallback)
# -------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_file, seq_len=20):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        df = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        required_imu = ["ax", "ay", "az", "gx", "gy", "gz"]
        required_cmd = ["linear_x", "angular_z"]
        
        for col in required_imu + required_cmd:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.imu = df[required_imu].values
        self.cmd = df[required_cmd].values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.imu) - self.seq_len

    def __getitem__(self, idx):
        seq = self.imu[idx:idx + self.seq_len]
        tgt = self.cmd[idx + self.seq_len]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)


# -------------------------------------------
# AUTO VERSIONING
# -------------------------------------------
def next_version_number(model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    versions = []

    for f in os.listdir(model_dir):
        match = re.match(r"lstm_v(\d+)\.pth", f)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1


# -------------------------------------------
# TRAINING LOOP (GPU Optimized)
# -------------------------------------------
def train_model(data_source="live", csv_path=None, epochs=20, seq_len=20, lr=0.001):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Select dataset
    if data_source == "live":
        print("📡 Using LIVE data from tunnel")
        dataset = LiveIMUDataset(seq_len=seq_len)
    else:
        print(f"📁 Using CSV data from: {csv_path}")
        dataset = CSVDataset(csv_path, seq_len=seq_len)
    
    # Check if we have enough data
    if len(dataset) == 0:
        print("❌ No training data available!")
        if data_source == "live":
            print("   Waiting for tunnel data... (run teleoperation to generate data)")
        return
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model and optimizer
    model = TeleopLSTM(input_size=6, hidden_size=64, num_layers=1, output_size=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"📊 Starting training with {len(dataset)} samples")
    print(f"📈 Sequence length: {seq_len}, Epochs: {epochs}")

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for seq, tgt in loader:
            # Move data to device
            seq = seq.to(device)
            tgt = tgt.to(device)
            
            pred = model(seq)
            loss = loss_fn(pred, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}  Samples: {len(dataset)}")

    # Save versioned model
    version = next_version_number()
    model_path = f"/app/models/lstm_v{version}.pth"
    torch.save(model.state_dict(), model_path)

    print(f"✅ Trained model saved: {model_path}")
    print(f"📋 Model specs: Input=6(IMU), Output=2(cmd_vel), Hidden=64, Layers=1")


# -------------------------------------------
# TUNNEL DATA COLLECTOR (Optional)
# -------------------------------------------
class TunnelDataCollector:
    def __init__(self, ngrok_url):
        self.ws = None
        self.ngrok_url = ngrok_url
        self.setup_websocket()
        
    def setup_websocket(self):
        """Connect to ngrok tunnel for data collection"""
        try:
            self.ws = websocket.WebSocketApp(self.ngrok_url,
                                            on_message=self.on_message,
                                            on_error=self.on_error,
                                            on_close=self.on_close)
            self.ws.on_open = self.on_open
            threading.Thread(target=self.ws.run_forever, daemon=True).start()
        except Exception as e:
            print(f"Tunnel connection failed: {e}")

    def on_message(self, ws, message):
        """Process incoming training data from tunnel"""
        try:
            data = json.loads(message)
            # You can process real-time training data here
            if data.get('type') == 'training_batch':
                print("Received training batch from tunnel")
        except Exception as e:
            print(f"Tunnel message error: {e}")

    def on_open(self, ws):
        print("✅ Tunnel connected for data collection")

    def on_error(self, ws, error):
        print(f"❌ Tunnel error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("⚠️ Tunnel data collection disconnected")


# -------------------------------------------
# MAIN with ROS integration
# -------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Parse arguments
    data_source = "live"  # Default to live tunnel data
    csv_path = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "csv" and len(sys.argv) > 2:
            data_source = "csv"
            csv_path = sys.argv[2]
        elif sys.argv[1] == "live":
            data_source = "live"
    
    try:
        print("🤖 Starting LSTM Training for IMU → cmd_vel mapping")
        print("==================================================")
        
        # Start training
        train_model(
            data_source=data_source,
            csv_path=csv_path,
            epochs=20,
            seq_len=20,
            lr=0.001
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        print("Training session ended")
