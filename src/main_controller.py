#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import json
import os
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import String
import time
import glob

# -------------------------------------------
# LSTM MODEL (Same as your trainer)
# -------------------------------------------
class TeleopLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, output_size=2):
        """ IMU: ax, ay, az, gx, gy, gz → linear_x, angular_z """
        super(TeleopLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class LSTMController:
    def __init__(self):
        print("=" * 60)
        print("🤖 LSTM AI Controller Initializing...")
        print("=" * 60)
        
        # GPU detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 Using device: {self.device}")
        
        # ROS setup
        rospy.init_node('lstm_controller', anonymous=True)
        
        # === PUBLISHER ===
        # Publish AI-generated commands that C++ controller will pick up
        self.cmd_pub = rospy.Publisher('/cmd_vel_ai', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/ai_status', String, queue_size=10)
        
        # === SUBSCRIBER ===
        # Listen to IMU data for LSTM inference
        self.imu_sub = rospy.Subscriber('/august/imu/data', Imu, self.imu_callback)
        
        # Find and load the latest trained LSTM model
        self.lstm_model = None
        self.load_latest_model()
        
        # IMU buffer for LSTM sequence
        self.imu_buffer = []
        self.sequence_length = 20  # Same as trainer
        self.max_buffer = 100
        
        # Control parameters
        self.control_rate = 10  # Hz
        self.last_control_time = 0
        
        # Scaling factors (tune these based on your training)
        self.linear_scale = 1.0
        self.angular_scale = 1.0
        
        print(f"\n✅ LSTM Controller Ready")
        print(f"📈 Sequence length: {self.sequence_length}")
        print(f"📡 Publishing to: /cmd_vel_ai")
        print(f"👂 Listening to: /august/imu/data")
        print("=" * 60)
    
    def load_latest_model(self):
        """Load the latest trained LSTM model from /app/models"""
        try:
            model_dir = "/app/models"
            if not os.path.exists(model_dir):
                model_dir = "/model"
                
            # Find all trained LSTM models
            lstm_files = glob.glob(f"{model_dir}/lstm_v*.pth")
            if not lstm_files:
                print("❌ No trained LSTM models found!")
                print(f"   Check directory: {model_dir}")
                print("   Run trainer first: python trainer.py csv /path/to/data.csv")
                return
            
            # Get the latest version
            lstm_files.sort(key=lambda x: int(x.split('_v')[-1].split('.pth')[0]))
            latest_model = lstm_files[-1]
            
            print(f"📦 Loading trained model: {latest_model}")
            
            # Load model architecture
            self.lstm_model = TeleopLSTM(
                input_size=6,
                hidden_size=64,
                num_layers=1,
                output_size=2
            )
            
            # Load trained weights
            self.lstm_model.load_state_dict(torch.load(latest_model, map_location=self.device))
            self.lstm_model.to(self.device)
            self.lstm_model.eval()
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"   Input: 6 (IMU data)")
            print(f"   Output: 2 (linear_x, angular_z)")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.lstm_model = None
    
    def imu_callback(self, imu_msg):
        """Process IMU data and generate LSTM predictions"""
        # Don't process if no model loaded
        if self.lstm_model is None:
            return
        
        # Extract IMU features (same as trainer expects)
        imu_features = [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ]
        
        # Add to buffer
        self.imu_buffer.append(imu_features)
        
        # Keep buffer size manageable
        if len(self.imu_buffer) > self.max_buffer:
            self.imu_buffer.pop(0)
        
        # Check if we have enough data for a sequence
        if len(self.imu_buffer) >= self.sequence_length:
            # Throttle control rate
            current_time = time.time()
            if current_time - self.last_control_time < 1.0 / self.control_rate:
                return
            
            self.last_control_time = current_time
            
            # Get the latest sequence
            sequence = self.imu_buffer[-self.sequence_length:]
            
            # Convert to tensor
            sequence_tensor = torch.tensor([sequence], dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.lstm_model(sequence_tensor)
            
            # Extract and scale predictions
            linear_x = prediction[0, 0].item() * self.linear_scale
            angular_z = prediction[0, 1].item() * self.angular_scale
            
            # Apply limits
            linear_x = np.clip(linear_x, -0.5, 0.5)
            angular_z = np.clip(angular_z, -1.0, 1.0)
            
            # Publish control command
            self.publish_control(linear_x, angular_z)
    
    def publish_control(self, linear_x, angular_z):
        """Publish AI-generated control command"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.linear.y = 0.0  # LSTM only predicts x and yaw
        cmd.linear.z = 0.2  # Small constant thrust (adjust as needed)
        cmd.angular.z = angular_z
        
        # Publish to /cmd_vel_ai (C++ controller picks this up)
        self.cmd_pub.publish(cmd)
        
        # Publish status
        status_msg = String()
        status_data = {
            'type': 'lstm_control',
            'linear_x': linear_x,
            'angular_z': angular_z,
            'timestamp': time.time(),
            'model_loaded': self.lstm_model is not None
        }
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)
        
        print(f"🤖 LSTM Control: X:{linear_x:.3f} Yaw:{angular_z:.3f}")
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        print("\n🚀 LSTM Controller Running...")
        print("   Model: Loaded" if self.lstm_model else "   Model: NOT LOADED (check /app/models)")
        print("   Publishing AI commands to: /cmd_vel_ai")
        print("   C++ controller will convert to motor commands")
        print("\nPress Ctrl+C to stop")
        
        while not rospy.is_shutdown():
            try:
                rate.sleep()
            except Exception as e:
                print(f"Main loop error: {e}")
                break

if __name__ == "__main__":
    try:
        controller = LSTMController()
        controller.run()
    except rospy.ROSInterruptException:
        print("\n🛑 LSTM Controller Shutdown")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")