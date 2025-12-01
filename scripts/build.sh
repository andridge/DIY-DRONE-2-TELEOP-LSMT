#!/bin/bash
# scripts/build.sh - CORRECTED for Noetic

echo "Building base image (Ubuntu 20.04 + ROS Noetic)..."
docker build -f Dockerfile.base -t drone-ai-base:latest .

echo "Building development image (UAT with tunnel support)..."
docker build -f Dockerfile.dev -t drone-ai-dev:latest .

echo "Building production image (with LoRa bridge)..."
docker build -f Dockerfile.prod -t drone-ai-prod:latest .

echo "All images built successfully!"
echo "UAT (with tunnel): bash scripts/run_dev.sh"
echo "Production (LoRa): bash scripts/run_prod.sh"
