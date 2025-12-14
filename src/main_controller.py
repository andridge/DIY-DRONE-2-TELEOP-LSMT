#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import json
import os
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import time
import glob
import re

try:
    import roslibpy
except ImportError:
    roslibpy = None


class TeleopLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=2):
        super(TeleopLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMController:
        def __init__(self):
            print("=" * 60)
            print("🤖 LSTM AI Controller Initializing...")
            print("=" * 60)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"📊 Using device: {self.device}")

            self.use_tunnel = False
            self.tunnel_url = os.environ.get("THECONSTRUCT_WS_URL", "").strip()
            if self.tunnel_url and roslibpy:
                self.use_tunnel = True
                print(f"🔌 Using tunnel via rosbridge: {self.tunnel_url}")
            elif self.tunnel_url and not roslibpy:
                print("❌ THECONSTRUCT_WS_URL set but roslibpy not installed; falling back to local ROS.")
                self.use_tunnel = False

            rospy.init_node('lstm_controller', anonymous=True)

            # Publish to /cmd_vel (mixer subscribes here)
            self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
            self.status_pub = rospy.Publisher('/ai_status', String, queue_size=10)

            # Subscribe to IMU and joint_states (talker topics)
            if not self.use_tunnel:
                self.imu_sub = rospy.Subscriber('/august/imu/data', Imu, self.imu_callback)
                self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
            else:
                self.setup_rosbridge_clients()

            # Buffers
            self.imu_buffer = []
            self.sequence_length = 20
            self.max_buffer = 200
            self.control_rate = 10
            self.last_control_time = 0

            # Latest IMU + joint positions
            self.last_imu = [0.0] * 6
            self.joints = {
                "revolute_47_joint": 0.0,
                "revolute_48_joint": 0.0,
                "revolute_49_joint": 0.0,
                "revolute_50_joint": 0.0,
            }

            # Scales
            self.linear_scale = 1.0
            self.angular_scale = 1.0

            # Model
            self.lstm_model = None
            # 10 features: ax, ay, az, gx, gy, gz, j47, j48, j49, j50
            self.input_size = 10
            self.load_latest_model()

            print(f"\n✅ LSTM Controller Ready")
            print(f"📈 Sequence length: {self.sequence_length}")
            print(f"👂 Listening: /august/imu/data + /joint_states")
            print(f"📡 Publishing: /cmd_vel")
            print("=" * 60)

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
            print("✅ Rosbridge subscriptions: /august/imu/data, /joint_states; publisher: /cmd_vel")

        # Rosbridge callbacks (dict payloads)
        def imu_callback_rb(self, msg):
            class Dummy:
                pass
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
            class DummyJ:
                pass
            js = DummyJ()
            js.name = msg.get('name', [])
            js.position = msg.get('position', [])
            self.joint_callback(js)

        def joint_callback(self, msg: JointState):
            # Keep latest relevant joint positions
            for i, name in enumerate(msg.name):
                if name in self.joints and i < len(msg.position):
                    self.joints[name] = msg.position[i]

        def load_latest_model(self):
            model_dir = "/app/models"
            candidates = glob.glob(f"{model_dir}/lstm_*.pth")
            # Also allow the existing file name you have
            fusion = os.path.join(model_dir, "lstm_fusion.pth")
            if os.path.exists(fusion):
                candidates.append(fusion)
            if not candidates:
                print("❌ No trained LSTM models found in /app/models!")
                return

            latest = max(candidates, key=os.path.getctime)
            print(f"📦 Loading: {latest}")

            try:
                data = torch.load(latest, map_location=self.device)

                if not isinstance(data, dict):
                    # Full model object
                    self.lstm_model = data.to(self.device)
                    self.lstm_model.eval()
                    print("✅ Full model object loaded")
                    return

                # checkpoint/state_dict handling
                state_dict = None
                if 'state_dict' in data:
                    state_dict = data['state_dict']
                elif 'model_state_dict' in data:
                    state_dict = data['model_state_dict']
                else:
                    state_dict = data

                # strip 'module.' prefix
                new_sd = {}
                for k, v in state_dict.items():
                    nk = k[7:] if k.startswith('module.') else k
                    new_sd[nk] = v
                state_dict = new_sd

                # Detect input_size from weight_ih if possible
                input_size = self.input_size
                for k, v in state_dict.items():
                    if 'weight_ih' in k:
                        try:
                            input_size = v.shape[1]
                        except Exception:
                            pass
                        break

                self.lstm_model = TeleopLSTM(input_size=input_size, output_size=2).to(self.device)
                self.lstm_model.load_state_dict(state_dict, strict=False)
                self.lstm_model.eval()
                print(f"✅ Model state_dict loaded (input_size={input_size})")
            except Exception as e:
                print(f"❌ Load failed: {e}")
                import traceback
                traceback.print_exc()
                self.lstm_model = None

        def imu_callback(self, imu_msg: Imu):
            # Update latest IMU values (6 DOF)
            self.last_imu = [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ]

            if not self.lstm_model:
                return

            # Build 10-feature vector
            features = [
                *self.last_imu,
                self.joints["revolute_47_joint"],
                self.joints["revolute_48_joint"],
                self.joints["revolute_49_joint"],
                self.joints["revolute_50_joint"],
            ]

            # Append to buffer
            self.imu_buffer.append(features)
            if len(self.imu_buffer) > self.max_buffer:
                self.imu_buffer.pop(0)

            # Rate-limit predictions
            if len(self.imu_buffer) >= self.sequence_length:
                now = time.time()
                if now - self.last_control_time < 1.0 / self.control_rate:
                    return
                self.last_control_time = now

                seq = torch.tensor([self.imu_buffer[-self.sequence_length:]],
                                   dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    pred = self.lstm_model(seq)[0]

                # 2 outputs: linear_x, angular_z
                linear_x = float(np.clip(pred[0].item() * self.linear_scale, -0.5, 0.5))
                angular_z = float(np.clip(pred[1].item() * self.angular_scale, -1.0, 1.0))
                # keep linear.y, linear.z conservative defaults (mixer handles thrust mixing)
                self.publish_control(linear_x, angular_z)

        def publish_control(self, linear_x, angular_z):
            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.linear.y = 0.0
            cmd.linear.z = 0.2  # small ascent bias; mixer clamps motor commands
            cmd.angular.z = angular_z

            if self.use_tunnel and hasattr(self, "rb_cmd"):
                self.rb_cmd.publish({
                    'linear': {'x': cmd.linear.x, 'y': cmd.linear.y, 'z': cmd.linear.z},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': cmd.angular.z}
                })
            else:
                self.cmd_pub.publish(cmd)

            status = String()
            status.data = json.dumps({
                "type": "lstm_control",
                "linear_x": round(linear_x, 3),
                "angular_z": round(angular_z, 3),
                "model_loaded": self.lstm_model is not None
            })
            self.status_pub.publish(status)

            print(f"🤖 AI→/cmd_vel: X:{linear_x:+.3f}  Z:{cmd.linear.z:+.3f}  Yaw:{angular_z:+.3f}")

        def run(self):
            print("\n🚀 LSTM Controller Running...")
            if self.use_tunnel:
                try:
                    while not rospy.is_shutdown():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            else:
                rospy.spin()


if __name__ == "__main__":
    try:
        controller = LSTMController()
        controller.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")