# Project Title

AI Misinformation Detection System

## Description

The backend API for misinformation detection system. Add the API keys for OPEN_AI and SERP_AI.

## Getting Started

- docker build -t irex4 .
- docker run -dp 5000:5000 -w /app -v "$(pwd):/app" irex4 sh -c "python main.py"
- docker exec -it <container_id> bash
