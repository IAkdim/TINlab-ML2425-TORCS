#!/bin/bash

# Debug wrapper script to run SCR client with logging from anywhere
# Usage: ./debug-client.sh [client-args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$SCRIPT_DIR/scr-client-cpp"
DEBUG_SCRIPT="$CLIENT_DIR/debug-client.sh"

# Check if debug script exists
if [ ! -f "$DEBUG_SCRIPT" ]; then
    echo "Error: debug script not found at $DEBUG_SCRIPT"
    exit 1
fi

echo "Running SCR client with debug logging..."
echo "Arguments: $@"
echo ""

# Run debug script from client directory
cd "$CLIENT_DIR"
"$DEBUG_SCRIPT" "$@"