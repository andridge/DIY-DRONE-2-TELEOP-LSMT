#!/bin/bash
# scripts/run_prod.sh - Production with LoRa

docker run -d --name drone-ai-prod --privileged -v /dev:/dev \
  -p 5678:5678 -p 8080:8080 -p 5000:5000 \
  -v $(pwd)/models:/app/models -v $(pwd)/config:/app/config -v $(pwd)/logs:/app/logs \
  -e DEPLOYMENT_ENV=production -e OPERATION_MODE=production -e LORA_PORT=/dev/ttyUSB0 \
  --restart unless-stopped drone-ai-prod:latest

echo "Production Container started with LoRa bridge!"
echo "   n8n → http://localhost:5678"
echo "   Dashboard → http://localhost:8080"
echo "   LoRa port: /dev/ttyUSB0"
