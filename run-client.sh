#!/bin/bash

# Wrapper script to run SCR client from anywhere
# Usage: ./run-client.sh [client-args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$SCRIPT_DIR/scr-client-cpp"
CLIENT_PATH="$CLIENT_DIR/client"

# Check if client exists
if [ ! -f "$CLIENT_PATH" ]; then
    echo "Error: client executable not found at $CLIENT_PATH"
    echo "Building client..."
    cd "$CLIENT_DIR"
    make clean && make
    if [ $? -ne 0 ]; then
        echo "Failed to build client"
        exit 1
    fi
    echo "Client built successfully!"
fi

echo "Running SCR client..."
echo "Arguments: $@"
echo ""

# Run client from its directory
cd "$CLIENT_DIR"
"$CLIENT_PATH" "$@"