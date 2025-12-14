# Drone AI Teleoperation – Operations Manual

> **IMPORTANT**
>
> * This is a **manual / runbook**.
> * **No commands have been deleted or modified**.
> * Commands are presented **exactly as provided**, with added headings and explanations only.
> * Follow sections in order unless troubleshooting.

---

## 1. Project Scripts

These scripts are used to build and run the system in different modes.

```bash
bash scripts/run_prod.sh
bash scripts/run_dev.sh
bash scripts/build.sh
```

---

## 2. Navigate to Project Directory

Ensure you are inside the project root before running Docker commands.

```bash
cd ~/Desktop/drone_ai_teleoperation
```

---

## 3. Docker Cleanup (Safe Reset)

Remove unused Docker resources and specific images.

```bash
docker system prune -f
docker rmi drone-ai-base drone-ai-dev || true
```

---

## 4. Build Base Docker Image

Build the base image used by all environments.

```bash
docker build --network=host -f Dockerfile.base -t drone-ai-base:latest .
```

---

## 5. Install ngrok (Linux)

Used for exposing local services (e.g., The Construct WebSocket).

```bash
wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
sudo tar xzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin/
rm ngrok-v3-stable-linux-amd64.tgz

ngrok --version
``` 

---

## 6. Model Training (Inside Container)

Run training jobs inside the **drone-ai-dev** container.

```bash
docker exec drone-ai-dev python3 /app/src/train_model.py live
docker exec drone-ai-dev python3 /app/src/train_model.py csv /app/models/imu_training_data.csv
```

---

## 7. GPU & CUDA Verification

Verify PyTorch and CUDA inside the base image.

```bash
docker run --gpus all drone-ai-base:latest python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## 8. Container Monitoring & Logs

### Check running containers

```bash
docker ps
```

### Supervisor services

```bash
docker exec drone-ai supervisorctl status
```

### Container logs

```bash
docker logs drone-ai
```

---

## 9. Run Production Container

Runs the AI system with GPU support and persistent volumes.

```bash
docker run -d --name drone-ai \
  --gpus all \
  -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  --restart unless-stopped \
  drone-ai-base:latest
```

---

## 10. Run Test Container (No GPU)

```bash
  docker run -d --name drone-ai-test \
  -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  drone-ai-base:latest
```

---

## 11. Low-Level GPU Validation (nvidia-smi)

```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

---

## 12. Restart Development Container

```bash
docker stop drone-ai-dev
docker rm drone-ai-dev
docker run -d --name drone-ai-dev \
  --gpus all \
  -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  -e THECONSTRUCT_WS_URL="ws://6.tcp.eu.ngrok.io:13793" \
  drone-ai-base:latest
```

---

## 13. GPU Check Inside Dev Container

```bash
docker exec drone-ai-dev python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU Detected:', torch.cuda.get_device_name(0))
    print('✓ CUDA Version:', torch.version.cuda)
    print('✓ GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
else:
    print('✗ No GPU detected')
"
```

---

## 14. Install Ultralytics YOLOv8

```bash
docker exec drone-ai-dev pip3 install ultralytics==8.0.196

docker exec drone-ai-dev pip3 install ultralytics==8.0.196
```

---

## 15. Test YOLOv8 GPU Access

```bash
# Test YOLOv8 with GPU
docker exec drone-ai-dev python3 -c "
from ultralytics import YOLO
import torch
print('Ultralytics installed')
print('PyTorch device:', torch.device('cuda'
"
```

---

## 16. Supervisor Status

```bash
docker exec drone-ai-dev supervisorctl status
```

---

## 17. Run Main Controller Manually

```bash
# Run main_controller.py manually in the container
docker exec -it drone-ai bash -c "source /opt/ros/noetic/setup.bash && python3 /app/src/main_controller.py"
```

---

## 18. Update main_controller.py (Hot Reload)

### Copy file locally

```bash
# Just copy the new file into the running container
cp main_controller.py ~/Desktop/drone_ai_teleoperation/src/main_controller.py
```

### Overwrite inside container

```bash
# Then overwrite inside the container (container name may vary)
docker cp src/main_controller.py drone-ai-dev:/app/src/main_controller.py
```

### Restart only AI node

```bash
# Restart only the AI node (no container restart!)
docker exec drone-ai-dev supervisorctl restart ai_pipeline
```

```bash
docker cp ~/Desktop/drone_ai_teleoperation/src/main_controller.py drone-ai-dev:/app/src/main_controller.py
```

---

## 19. Inspect Model File (LSTM)

```bash
# Check model file size and info
docker exec drone-ai-dev python3 -c "
import torch
model_path = '/app/models/lstm_fusion.pth'
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print('Model keys:', checkpoint.keys())
    
    # Check LSTM weight dimensions
    for key, value in checkpoint.items():
        if 'weight' in key or 'bias' in key:
            print(f'{key}: shape {tuple(value.shape)}')
            
    # Try to load and inspect
    print('\\nTrying to load...')
    model = torch.load(model_path, map_location='cpu')
    print('Model loaded successfully')
    
except Exception as e:
    print(f'Error: {e}')
"
```

---

## 20. Full Cleanup & Rebuild (Nuclear Option)

### Stop and remove containers

```bash
# Stop and remove containers (by name or id)
docker stop drone-ai-dev nervous_mahavira || true
docker rm -f drone-ai-dev nervous_mahavira || true
```

### Remove image

```bash
# Remove the custom image (ignore errors)
docker rmi drone-ai-base:latest || true
```

### Aggressive cleanup

```bash
# Optional: aggressive cleanup (removes all unused images, containers, networks, volumes)
# WARNING: may remove volumes with data
docker system prune -af --volumes
```

---

## 21. Rebuild and Start Fresh Dev Environment

```bash
# Rebuild the base image
cd ~/Desktop/drone_ai_teleoperation
docker build --network=host -f Dockerfile.base -t drone-ai-base:latest .
```

```bash
# Start a fresh dev container (adjust env / ports / mounts as needed)
docker run -d --name drone-ai-dev \
  --gpus all \
  -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  -e THECONSTRUCT_WS_URL="ws://6.tcp.eu.ngrok.io:13793" \
  drone-ai-base:latest
```

---

## 22. Final Verification

```bash
# Verify
docker ps
docker logs -f drone-ai-dev
```

## 21. Copy updated trainer into the container
docker cp src/train_lstm_imu.py drone-ai:/app/src/train_lstm_imu.py

## 24. Run live training inside the container
docker exec -it drone-ai bash -c "source /opt/ros/noetic/setup.bash && python3 /app/src/train_lstm_imu.py live"

# list models to confirm the new file
docker exec drone-ai-dev ls -la /app/models


########################################

docker exec -it drone-ai bash -c '\
  export THECONSTRUCT_WS_URL="tcp://2.tcp.eu.ngrok.io:10417"; \
  source /opt/ros/noetic/setup.bash; \
  python3 /app/src/main_controller.py'


  docker exec -it drone-ai bash -c '\
  export THECONSTRUCT_WS_URL="tcp://2.tcp.eu.ngrok.io:10417"; \
  source /opt/ros/noetic/setup.bash; \
  python3 /app/src/train_lstm_imu.py live'