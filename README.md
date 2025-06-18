# TORCS SCR (Simulated Car Racing) Setup

This repository contains a complete setup for running TORCS with the SCR (Simulated Car Racing) patch, which allows external AI clients to connect and control virtual race cars via UDP networking.

## What's Included

- **TORCS 1.3.4** with SCR patch applied
- **Docker containers** for easy deployment
- **SCR C++ client** with SimpleDriver implementation
- **headless operation** using Xvfb virtual display

## What's not included
- Full TORCS source code (download 1.3.4 from sourceforge and drop the tarball here)
- SCR patch (download from the SCR GitHub repository and drop the tarball here)

## Quick Start

### 1. Build the Docker Images

```bash
# Build the TORCS container (includes Xvfb for headless operation)
docker build -f Dockerfile -t torcs-server .
```

### 2. Build the SCR Client

```bash
cd scr-client-cpp
make clean && make
```

### 3. Run a Test Race

#### Terminal 1: Start the TORCS Server
```bash
# Start server with UDP port 3001 exposed
docker run --rm -d -p 3001:3001/udp --name torcs-server torcs-server:latest

# Wait for server initialization
sleep 10
echo "TORCS server is ready!"
```

#### Terminal 2: Connect the SCR Client
```bash
cd scr-client-cpp

# Run a single race with 2000 steps on aalborg track
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:2000 track:aalborg
```

#### Stop the Server
```bash
docker stop torcs-server
```

## Configuration Files

### Race Configuration (`practice.xml`)
```xml
<!-- practice.xml -->
<params name="Practice Race">
  <section name="Tracks">
    <attstr name="name" val="aalborg"/>
    <attstr name="category" val="road"/>
  </section>

  <section name="Drivers">
    <section name="1">
      <attnum name="idx" val="1"/>
      <attstr name="module" val="scr_server"/>  <!-- Uses SCR server -->
    </section>
  </section>

  <section name="Race">
    <attnum name="distance" unit="km" val="0"/>
    <attnum name="laps" val="1"/>
    <attstr name="display mode" val="results only"/>
  </section>
</params>
```

## Client Parameters

The SCR client accepts these command-line parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `host` | Server hostname | localhost | `host:192.168.1.100` |
| `port` | Server UDP port | 3001 | `port:3001` |
| `id` | Client identifier | SCR | `id:MyBot` |
| `maxEpisodes` | Number of races | 1 | `maxEpisodes:5` |
| `maxSteps` | Steps per race | 100000 | `maxSteps:2000` |
| `track` | Track name | unknown | `track:aalborg` |
| `stage` | Race stage | UNKNOWN | `stage:PRACTICE` |

### Example Client Commands

```bash
# Basic test run
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:1000

# Multiple episodes
./client host:localhost port:3001 id:SCR maxEpisodes:3 maxSteps:5000

# Specific track
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:2000 track:aalborg

# Long race
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:10000 track:corkscrew
```

## Available Tracks

Common tracks included in TORCS:
- `aalborg` - Road track
- `corkscrew` - Road track with elevation changes
- `forza` - Oval track
- `michigan` - Speedway
- `wheel-1` - Road course

You can list all available tracks:
```bash
docker run --rm torcs-fixed:latest find /usr/local/share/games/torcs/tracks -name "*.xml" | grep -v "\.acc" | sed 's/.*tracks\/[^\/]*\///; s/\/.*\.xml//' | sort | uniq
```

## Test Scenarios

### 1. Quick Connection Test
```bash
# Start server
docker run --rm -d -p 3001:3001/udp --name torcs-server torcs-fixed:latest
sleep 10

# Quick 30-second test
timeout 30 ./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:100

# Stop server
docker stop torcs-server
```

### 2. Full Race Test
```bash
# Start server
docker run --rm -d -p 3001:3001/udp --name torcs-server torcs-fixed:latest
sleep 10

# Run complete race (will finish when car completes track or crashes)
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:50000 track:aalborg

# Stop server
docker stop torcs-server
```

### 3. Multiple Episodes Test
```bash
# Start server
docker run --rm -d -p 3001:3001/udp --name torcs-server torcs-fixed:latest
sleep 10

# Run 3 races of 10000 steps each
./client host:localhost port:3001 id:SCR maxEpisodes:3 maxSteps:10000 track:aalborg

# Stop server
docker stop torcs-server
```

### 4. Performance Benchmark
```bash
# Start server
docker run --rm -d -p 3001:3001/udp --name torcs-server torcs-fixed:latest
sleep 10

# Time a race
time ./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:5000 track:aalborg

# Stop server
docker stop torcs-server
```

## Client Output Explanation

When you run the client, you'll see output like:
```
***********************************
HOST: localhost
PORT: 3001
ID: SCR
MAX_STEPS: 2000
MAX_EPISODES: 1
TRACKNAME: aalborg
STAGE: UNKNOWN
***********************************
Sending id to server: SCR
Sending init string to the server: SCR(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)
Received: ***identified***
Restarting the race!
Client Restart
Bye bye!
```

This indicates:
1. ✅ **Connection successful** - Client connected to server
2. ✅ **Identification accepted** - Server recognized the client
3. ✅ **Race initialized** - Race started successfully
4. ✅ **Race completed** - Client finished and disconnected

## Troubleshooting

### Server Won't Start
```bash
# Check if port is already in use
netstat -ulnp | grep 3001

# Kill existing processes
sudo pkill -f torcs
```

### Client Can't Connect
```bash
# Check if server container is running
docker ps | grep torcs-server

# Check server logs
docker logs torcs-server

# Test port connectivity
nc -u localhost 3001
```

### Performance Issues
```bash
# Monitor Docker container resources
docker stats torcs-server

# Run with more verbose output
docker run --rm -p 3001:3001/udp torcs-fixed:latest  # Run in foreground
```

## Development

### Modifying the Client
The SimpleDriver implementation is in `scr-client-cpp/SimpleDriver.cpp`. You can modify the driving logic and rebuild:

```bash
cd scr-client-cpp
# Edit SimpleDriver.cpp
make clean && make
```

### Custom Race Configurations
Create custom practice files by modifying `practice.xml`:
- Change track: `<attstr name="name" val="your-track"/>`
- Change laps: `<attnum name="laps" val="5"/>`
- Change distance: `<attnum name="distance" unit="km" val="10"/>`

### Network Configuration
The SCR server uses these default network settings:
- **Protocol**: UDP
- **Port**: 3001 (configurable in source)
- **Client ID**: "SCR"
- **Max concurrent clients**: 10 (ports 3001-3010)

## File Structure

```
torcs/
├── README.md                 # This file
├── Dockerfile.fixed          # Fixed Docker build with Xvfb
├── practice.xml             # Race configuration
├── torcs-1.3.4.tar.bz2     # Original TORCS source
├── scr-linux-patch.tgz     # SCR patch
└── scr-client-cpp/          # SCR client implementation
    ├── client.cpp           # Main client code
    ├── SimpleDriver.cpp     # AI driver implementation
    ├── SimpleDriver.h
    ├── CarState.cpp         # Car sensor data
    ├── CarControl.cpp       # Car control commands
    └── Makefile            # Build configuration
```

## Technical Details

### Fixed Issues
- ✅ **Segmentation fault** - Fixed by adding Xvfb virtual display
- ✅ **Missing libraries** - Fixed LD_LIBRARY_PATH in startup script
- ✅ **Headless operation** - Added virtual framebuffer support
- ✅ **Network communication** - SCR UDP protocol working correctly

### System Requirements
- Docker
- Linux/macOS (Windows with WSL2)
- Available UDP port 3001
- ~2GB disk space for Docker images

### Protocol Information
The SCR protocol uses UDP messages with this format:
- **Client → Server**: Control commands (steering, acceleration, etc.)
- **Server → Client**: Sensor data (speed, position, track sensors, etc.)
- **Message format**: Plain text with parenthetical structure

## Viewing Server Responses (Sensor Data)

The SCR client now includes verbose output to see all sensor data from the server. You can monitor the communication in several ways:

### 1. Built-in Verbose Mode
The client has been modified to show all server responses:

```bash
# Each response shows complete sensor data
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:100
```

**Example Server Response:**
```
Received: (angle 0.00349682)(curLapTime -0.982)(damage 0)(distFromStart 1957.56)(distRaced 0)(fuel 94)(gear 0)(lastLapTime 0)(opponents 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)(racePos 1)(rpm 942.478)(speedX 0.00190964)(speedY 0.0500837)(speedZ -0.00538569)(track 7.4866 7.79257 8.84572 11.3663 20.0191 64.5828 55.6697 47.273 39.6595 33.0475 27.546 23.1283 19.6619 16.9697 13.2465 10.125 8.52485 7.74516 7.5135)(trackPos 0.00179443)(wheelSpinVel 0 0 0 0)(z 0.346284)(focus -1 -1 -1 -1 -1)
```

### 2. Debug Script with Logging
Use the debug script to save all communication to a file:

```bash
# Run with automatic logging
./debug-client.sh host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:1000

# View only sensor data
grep 'Received:' scr_debug_*.log

# View only car state (excluding control messages)
grep 'Received:' scr_debug_*.log | grep -v 'identified\|restart\|shutdown'
```

### 3. Real-time Monitoring
Monitor specific sensor values in real-time:

```bash
# Monitor speed and RPM
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:1000 | grep -o 'rpm [0-9.]*\|speedX [0-9.-]*'

# Monitor track sensors (distance to track edges)
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:1000 | grep -o 'track [0-9. ]*'

# Monitor steering angle and position
./client host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:1000 | grep -o 'angle [0-9.-]*\|trackPos [0-9.-]*'
```

## Sensor Data Explanation

The server sends sensor data in this format: `(sensor_name value)`

| Sensor | Description | Range/Units |
|--------|-------------|-------------|
| `angle` | Car's angle to track direction | radians (-π to π) |
| `curLapTime` | Current lap time | seconds |
| `damage` | Car damage level | 0-10000 |
| `distFromStart` | Distance from start line | meters |
| `distRaced` | Total distance raced | meters |
| `fuel` | Remaining fuel | liters |
| `gear` | Current gear | -1,0,1,2,3,4,5,6 |
| `lastLapTime` | Previous lap time | seconds |
| `opponents` | Distance to other cars | meters (36 values) |
| `racePos` | Position in race | 1st, 2nd, etc. |
| `rpm` | Engine RPM | revolutions/minute |
| `speedX` | Speed along track | m/s |
| `speedY` | Lateral speed | m/s |
| `speedZ` | Vertical speed | m/s |
| `track` | Distance to track edges | meters (19 sensors) |
| `trackPos` | Position on track | -1 to 1 (left to right) |
| `wheelSpinVel` | Wheel spin velocities | rad/s (4 wheels) |
| `z` | Height above track | meters |
| `focus` | Focus sensors | meters (5 sensors) |

## Control Commands Sent to Server

The client sends control commands in this format: `(command value)`

| Command | Description | Range |
|---------|-------------|-------|
| `accel` | Acceleration pedal | 0.0-1.0 |
| `brake` | Brake pedal | 0.0-1.0 |
| `gear` | Gear selection | -1,0,1,2,3,4,5,6 |
| `steer` | Steering wheel | -1.0 to 1.0 |
| `clutch` | Clutch pedal | 0.0-1.0 |
| `focus` | Focus direction | -90 to 90 degrees |
| `meta` | Meta command | 0 or 1 |

## Advanced Debugging

### Analyze Driving Patterns
```bash
# Extract and analyze turning behavior
grep 'Received:' debug.log | grep -o 'steer [0-9.-]*\|angle [0-9.-]*' > steering_analysis.txt

# Track speed over time
grep 'Received:' debug.log | grep -o 'speedX [0-9.-]*\|rpm [0-9.]*' > speed_analysis.txt

# Monitor fuel consumption
grep 'Received:' debug.log | grep -o 'fuel [0-9]*' > fuel_analysis.txt
```

### Performance Metrics
```bash
# Count successful communications
grep -c 'Received:' debug.log

# Check for communication errors
grep 'didn\'t get response\|cannot send' debug.log

# Monitor race progress
grep 'distRaced' debug.log | tail -10
```

This setup provides a complete environment for developing and testing AI race car drivers using the TORCS SCR platform.
