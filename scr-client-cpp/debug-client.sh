#!/bin/bash

# Debug script for SCR client with verbose logging
# Usage: ./debug-client.sh [client-args...]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_PATH="$SCRIPT_DIR/client"

# Check if client exists
if [ ! -f "$CLIENT_PATH" ]; then
    echo "Error: client executable not found at $CLIENT_PATH"
    echo "Please run 'make' in the scr-client-cpp directory first."
    exit 1
fi

LOGFILE="scr_debug_$(date +%Y%m%d_%H%M%S).log"

echo "Starting SCR client with debug logging..."
echo "Output will be saved to: $LOGFILE"
echo "Press Ctrl+C to stop"
echo ""

# Run client with all output captured
"$CLIENT_PATH" "$@" 2>&1 | tee "$LOGFILE"

echo ""
echo "Debug session completed. Log saved to: $LOGFILE"
echo "To view sensor data:"
echo "  grep 'Received:' $LOGFILE"
echo "To view only car state:"
echo "  grep 'Received:' $LOGFILE | grep -v 'identified\\|restart\\|shutdown'"