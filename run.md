bash scripts/run_prod.sh
bash scripts/run_dev.sh
bash scripts/build.sh
cd ~/Desktop/drone_ai_teleoperation
docker system prune -f
docker rmi drone-ai-base drone-ai-dev || true

docker build --network=host -f Dockerfile.base -t drone-ai-base:latest .



wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
sudo tar xzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin/
rm ngrok-v3-stable-linux-amd64.tgz

ngrok --version

docker exec drone-ai-dev python3 /app/src/train_model.py live
docker exec drone-ai-dev python3 /app/src/train_model.py csv /app/models/imu_training_data.csv


docker run --gpus all drone-ai-base:latest python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"


# Check container status
docker ps

# Check supervisor services
docker exec drone-ai supervisorctl status

# Check logs
docker logs drone-ai

docker run -d --name drone-ai \
  --gpus all \
  -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  --restart unless-stopped \
  drone-ai-base:latest


  docker run -d --name drone-ai-test \
  -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  drone-ai-base:latest