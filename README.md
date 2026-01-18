Sensors + Camera (Drone)
        ↓
Ngrok Tunnel (Upstream)
        ↓
GPU System (ROS + LSTM + YOLO)
        ↓
Desired State Output:
- Roll rate
- Pitch rate
- Yaw rate
- Throttle
- Mode (takeoff / land / cruise)
- Constants (altitude, heading, etc.)
        ↓
Ngrok Tunnel (Downstream)
        ↓
Embedded System:
- Apply constants
- Convert rates → PWM
- Maintain stability
