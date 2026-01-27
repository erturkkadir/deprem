#!/bin/bash
# Auto-restart wrapper for earthquake prediction server
# Restarts server if it crashes

cd /var/www/syshuman/quake

while true; do
    echo "[$(date)] Starting server..."
    python server.py 2>&1 | tee -a server.out

    EXIT_CODE=$?
    echo "[$(date)] Server exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Clean exit, not restarting"
        break
    fi

    echo "[$(date)] Server crashed! Restarting in 5 seconds..."
    sleep 5
done
