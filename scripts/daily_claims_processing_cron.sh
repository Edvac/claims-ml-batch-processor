#!/bin/bash
# Simple cron job for daily claims processing

# Set working directory to project root, relative to script location
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Run daily batch processing 
python main.py --batch ./data/daily_claims/ --pattern "claims_*.csv" 2>&1 | tee ./logs/daily_$(date +%Y%m%d).log

# Check exit status and send notification if failed
if [ $? -ne 0 ]; then
    echo "Claims ML pipeline failed on $(date)" | mail -s "Claims Pipeline Failure" trevor.claims@company.com
fi