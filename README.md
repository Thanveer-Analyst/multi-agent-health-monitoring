# Multi-Agent Health Monitoring with DRL

This repository implements a multi-agent deep reinforcement learning framework for mapping vital‐sign measurements (e.g., heart-rate, respiration, temperature) to clinical escalation levels (MET-0…MET-4) using MEWS thresholds.

## Features
- **Custom Gym Environments** for each vital sign based on MEWS thresholds  
- **DQN Agents** with ε-greedy exploration and replay memory  
- Configurable hyperparameters (`EPISODES`, `LEARNING_RATE`, etc.)  
- Example training scripts for heart-rate, respiration, and temperature  

## Setup

1. **Clone** the repo
    git clone https://github.com/<your-username>/multi-agent-health-monitoring.git
    cd multi-agent-health-monitoring
2. Create & activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate
3. Install dependencies
    pip install -r requirements.txt