#!/bin/bash
# scripts/run_dev.sh - UAT with tunnel

set -e

echo "Enter your ngrok WebSocket URL (e.g. ws://7.tcp.eu.ngrok.io:12636):"
read -r NGROK_WS

echo "Launching AI Brain (UAT) → connecting to $NGROK_WS"

docker run -d --name drone-ai-dev \
  --gpus all \
  -p 5678:5678 -p 9090:9090 -p 8080:8080 -p 5000:5000 \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  -e OPERATION_MODE=uat \
  -e THECONSTRUCT_WS_URL="$NGROK_WS" \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=droneai2025 \
  --restart unless-stopped \
  drone-ai-dev:latest

echo "UAT Container started! Access:"
echo "   n8n → http://localhost:5678   (user: admin / pass: droneai2025)"
echo "   ROSbridge → ws://localhost:9090"
echo "   Dashboard → http://localhost:8080"
echo "   Connected to tunnel: $NGROK_WS"
