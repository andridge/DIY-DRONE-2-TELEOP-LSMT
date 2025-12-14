#!/usr/bin/env python3
import os
import re
import threading
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import rospy
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist

try:
    import roslibpy
except ImportError:
    roslibpy = None

# -------------------------------------------
# LSTM MODEL (GPU Enabled)
# -------------------------------------------
class TeleopLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, output_size=2):
        """10 inputs: ax,ay,az,gx,gy,gz,j47,j48,j49,j50 -> 2 outputs: linear_x, angular_z"""
        super(TeleopLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)
        print(f"Model initialized on: {self.device}")

    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# -------------------------------------------
# DATASET FOR LIVE ROS OR ROSBRIDGE DATA
# -------------------------------------------
class LiveIMUDataset(Dataset):
    def __init__(self, seq_len=20, max_samples=5000):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.imu_buffer = []   # [ax, ay, az, gx, gy, gz, j47, j48, j49, j50]
        self.cmd_buffer = []   # [linear_x, angular_z]
        self.joints = {
            "revolute_47_joint": 0.0,
            "revolute_48_joint": 0.0,
            "revolute_49_joint": 0.0,
            "revolute_50_joint": 0.0,
        }
        self.lock = threading.Lock()

        # Tunnel detection
        self.use_tunnel = False
        self.tunnel_url = os.environ.get("THECONSTRUCT_WS_URL", "").strip()
        if self.tunnel_url and roslibpy:
            self.use_tunnel = True
            print(f"🔌 Dataset via rosbridge tunnel: {self.tunnel_url}")
        elif self.tunnel_url and not roslibpy:
            print("❌ THECONSTRUCT_WS_URL set but roslibpy not installed; falling back to local ROS.")

        rospy.init_node('live_training_dataset', anonymous=True)

        if not self.use_tunnel:
            rospy.Subscriber('/august/imu/data', Imu, self.imu_callback)
            rospy.Subscriber('/joint_states', JointState, self.joint_callback)
            rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback)
        else:
            self.setup_rosbridge_clients()

        print("Live dataset: listening to /august/imu/data + /joint_states + /cmd_vel")

    def setup_rosbridge_clients(self):
        m = re.match(r"ws://([^:/]+):(\d+)", self.tunnel_url)
        if not m:
            print(f"❌ Invalid THECONSTRUCT_WS_URL: {self.tunnel_url}")
            self.use_tunnel = False
            return
        host, port = m.group(1), int(m.group(2))
        self.rb = roslibpy.Ros(host=host, port=port, is_secure=False)
        self.rb.run()
        self.rb_imu = roslibpy.Topic(self.rb, '/august/imu/data', 'sensor_msgs/Imu')
        self.rb_joints = roslibpy.Topic(self.rb, '/joint_states', 'sensor_msgs/JointState')
        self.rb_cmd = roslibpy.Topic(self.rb, '/cmd_vel', 'geometry_msgs/Twist')
        self.rb_imu.subscribe(self.imu_callback_rb)
        self.rb_joints.subscribe(self.joint_callback_rb)
        self.rb_cmd.subscribe(self.cmd_callback_rb)
        print("✅ Rosbridge dataset subscriptions active")

    # rosbridge -> rospy-style callback shims
    def imu_callback_rb(self, msg):
        class Dummy: pass
        imu = Dummy()
        imu.linear_acceleration = Dummy()
        imu.angular_velocity = Dummy()
        imu.linear_acceleration.x = msg['linear_acceleration']['x']
        imu.linear_acceleration.y = msg['linear_acceleration']['y']
        imu.linear_acceleration.z = msg['linear_acceleration']['z']
        imu.angular_velocity.x = msg['angular_velocity']['x']
        imu.angular_velocity.y = msg['angular_velocity']['y']
        imu.angular_velocity.z = msg['angular_velocity']['z']
        self.imu_callback(imu)

    def joint_callback_rb(self, msg):
        class DummyJ: pass
        js = DummyJ()
        js.name = msg.get('name', [])
        js.position = msg.get('position', [])
        self.joint_callback(js)

    def cmd_callback_rb(self, msg):
        class DummyC: pass
        c = DummyC()
        c.linear = type("V", (), {})()
        c.angular = type("V", (), {})()
        c.linear.x = msg['linear']['x']
        c.angular.z = msg['angular']['z']
        self.cmd_callback(c)

    # --- Native callbacks ---
    def joint_callback(self, msg):
        with self.lock:
            for i, name in enumerate(msg.name):
                if name in self.joints and i < len(msg.position):
                    self.joints[name] = msg.position[i]

    def imu_callback(self, msg):
        imu_vals = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]
        with self.lock:
            sample = imu_vals + [
                self.joints["revolute_47_joint"],
                self.joints["revolute_48_joint"],
                self.joints["revolute_49_joint"],
                self.joints["revolute_50_joint"],
            ]
            self.imu_buffer.append(sample)
            if len(self.imu_buffer) > self.max_samples:
                self.imu_buffer.pop(0)

    def cmd_callback(self, msg):
        cmd_vals = [msg.linear.x, msg.angular.z]
        with self.lock:
            self.cmd_buffer.append(cmd_vals)
            if len(self.cmd_buffer) > self.max_samples:
                self.cmd_buffer.pop(0)

    def __len__(self):
        with self.lock:
            return max(0, min(len(self.imu_buffer), len(self.cmd_buffer)) - self.seq_len)

    def __getitem__(self, idx):
        with self.lock:
            if idx + self.seq_len >= len(self.imu_buffer) or idx + self.seq_len >= len(self.cmd_buffer):
                return torch.zeros(self.seq_len, 10), torch.zeros(2)
            seq = self.imu_buffer[idx:idx + self.seq_len]
            tgt = self.cmd_buffer[idx + self.seq_len]
            return torch.tensor(seq, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# -------------------------------------------
# CSV DATASET (Fallback)
# -------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_file, seq_len=20):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        df = pd.read_csv(csv_file)
        required_imu = ["ax", "ay", "az", "gx", "gy", "gz", "j47", "j48", "j49", "j50"]
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
def next_version_number(model_dir="/app/models"):
    os.makedirs(model_dir, exist_ok=True)
    versions = []
    for f in os.listdir(model_dir):
        m = re.match(r"lstm_v(\d+)\.pth", f)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1

# -------------------------------------------
# TRAINING LOOP
# -------------------------------------------
def train_model(data_source="live", csv_path=None, epochs=20, seq_len=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if data_source == "live":
        dataset = LiveIMUDataset(seq_len=seq_len)
        for _ in range(5):
            rospy.sleep(1.0)
            if len(dataset) > 50:
                break
    else:
        dataset = CSVDataset(csv_path, seq_len=seq_len)

    if len(dataset) == 0:
        print("❌ No training data available.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TeleopLSTM(input_size=10, hidden_size=64, num_layers=1, output_size=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"📈 Samples: {len(dataset)}, Epochs: {epochs}, Seq: {seq_len}")
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for seq, tgt in loader:
            seq = seq.to(device)
            tgt = tgt.to(device)
            pred = model(seq)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"Epoch {epoch+1}/{epochs}  Loss: {total / max(1, n):.6f}")

    version = next_version_number()
    model_path = f"/app/models/lstm_v{version}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved: {model_path}  (input=10, output=2, hidden=64)")

# -------------------------------------------
if __name__ == "__main__":
    import sys
    data_source = "live"
    csv_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "csv" and len(sys.argv) > 2:
        data_source = "csv"
        csv_path = sys.argv[2]
    try:
        train_model(data_source=data_source, csv_path=csv_path, epochs=20, seq_len=20, lr=0.001)
    except KeyboardInterrupt:
        print("Interrupted")