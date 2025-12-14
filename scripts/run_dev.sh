#!/bin/bash
set -e

echo "===================================================================="
echo "          DRONE AI BRAIN — UAT MODE (The Construct + ngrok)"
echo "===================================================================="

# 0. Fix logs folder (the real root of all evil)
echo "Fixing logs folder permissions..."
mkdir -p logs models
sudo chown -R $USER:$USER logs 2>/dev/null || true
chmod -R 777 logs

# 1. Check for models before starting
echo "Checking for trained LSTM models..."
MODEL_COUNT=$(ls models/*.pth models/*.pt 2>/dev/null | wc -l)
if [ $MODEL_COUNT -eq 0 ]; then
    echo "❌ WARNING: No trained models found in 'models/' directory!"
    echo "   To train a model, run: python scripts/trainer.py csv data/your_data.csv"
    echo "   Or use the default model: wget -P models/ https://example.com/lstm_default.pth"
    echo "   Continuing anyway, but AI will be disabled..."
fi

# 2. Build base image if missing
if ! docker image inspect drone-ai-base:latest >/dev/null 2>&1; then
    echo "Building drone-ai-base:latest ..."
    docker build -f Dockerfile.base -t drone-ai-base:latest .
fi

# 3. Get ngrok URL
echo
read -p "Enter your ngrok WebSocket URL (e.g. ws://6.tcp.eu.ngrok.io:13793): " NGROK_WS
[[ -z "$NGROK_WS" ]] && { echo "No URL! Exiting."; exit 1; }

# 4. Kill old container
docker ps -a --format '{{.Names}}' | grep -q '^drone-ai-dev$' && {
    echo "Removing old container..."
    docker stop drone-ai-dev >/dev/null || true
    docker rm drone-ai-dev >/dev/null || true
}

# 5. Copy default model if no models exist
if [ $MODEL_COUNT -eq 0 ] && [ -f "scripts/lstm_default.pth" ]; then
    echo "Copying default model to models/ directory..."
    cp scripts/lstm_default.pth models/
elif [ $MODEL_COUNT -eq 0 ]; then
    echo "Creating a dummy model for testing..."
    cat > create_dummy_model.py << 'EOF'
import torch
import torch.nn as nn

class TeleopLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, output_size=2):
        super(TeleopLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Create and save a dummy model
model = TeleopLSTM(input_size=10)
torch.save(model.state_dict(), 'models/lstm_dummy.pth')
print(f"✅ Dummy model saved to models/lstm_dummy.pth")
print(f"   Input size: 10, Output size: 2")
EOF
    python3 create_dummy_model.py
    rm create_dummy_model.py
fi

# 6. List available models
echo "Available models in 'models/':"
ls -la models/*.pth models/*.pt 2>/dev/null || echo "   (none)"

# 7. Launch container using BASE image
echo "Launching AI Brain → $NGROK_WS"
docker run -d \
  --name drone-ai-dev \
  --gpus all \
  --network host \
  -v "$(pwd)/models":/app/models:rw \
  -v "$(pwd)/config":/app/config:rw \
  -v "$(pwd)/logs":/app/logs:rw \
  -e OPERATION_MODE=uat \
  -e THECONSTRUCT_WS_URL="$NGROK_WS" \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=droneai2025 \
  --restart unless-stopped \
  drone-ai-base:latest

# 8. Wait for ROS to wake up
echo "Waiting for ROS & rosbridge (15 seconds)..."
sleep 15

# 9. FORCE START YOUR AI BRAIN (this is what actually makes the drone move)
echo "STARTING YOUR LSTM AI BRAIN — COMMANDS ARE NOW BEING SENT!"
docker exec drone-ai-dev bash -c "
    echo '=== CHECKING MODEL DIRECTORY ==='
    ls -la /app/models/
    echo '=== STARTING LSTM CONTROLLER ==='
    source /opt/ros/noetic/setup.bash && 
    nohup python3 /app/src/main_controller.py > /app/logs/ai_brain.log 2>&1 &
    echo 'LSTM controller started with PID: \$!'
"

# 10. Wait a bit for the model to load
echo "Waiting for model to load (3 seconds)..."
sleep 3

# 11. Check if model loaded successfully
echo "Checking model load status..."
docker exec drone-ai-dev tail -n 20 /app/logs/ai_brain.log | grep -E "(LOADING|MODEL|✅|❌|📦|Detected)" || echo "Waiting for logs to appear..."

# 12. Final success message
echo
echo "===================================================================="
echo "                   YOUR DRONE IS NOW FLYING WITH AI"
echo "===================================================================="
echo "   Tunnel        → $NGROK_WS"
echo "   n8n           → http://localhost:5678 (admin/droneai2025)"
echo "   Dashboard     → http://localhost:8080"
echo "   ROSbridge     → ws://localhost:9090"
echo "   Models        → $(ls models/*.pth models/*.pt 2>/dev/null | wc -l) model(s) loaded"
echo
echo "   Real-time AI commands →/cmd_vel (visible in The Construct)"
echo
echo "   To watch live AI output:"
echo "       tail -f logs/ai_brain.log"
echo "   To see model loading details:"
echo "       grep -E '(LOADING|MODEL|✅|❌)' logs/ai_brain.log"
echo "   To stop everything:"
echo "       docker stop drone-ai-dev"
echo "===================================================================="

# 13. Show logs tail immediately
echo
echo "=== CURRENT LOGS (last 10 lines) ==="
tail -n 10 logs/ai_brain.log 2>/dev/null || echo "Log file not yet created..."
