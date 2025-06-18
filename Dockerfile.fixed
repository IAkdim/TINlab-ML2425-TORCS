FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies including xvfb for headless graphics
RUN apt-get update && apt-get install -y \
    build-essential \
    xorg-dev \
    libxi-dev \
    libxmu-dev \
    libx11-dev \
    libgl1-mesa-dev \
    libalut-dev \
    libvorbis-dev \
    libplib-dev \
    freeglut3-dev \
    libjpeg-dev \
    zlib1g-dev \
    patch \
    wget \
    unzip \
    git \
    cmake \
    libpng-dev \
    libopenal-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt

# Copy the source archive and patch
COPY ./torcs-1.3.4.tar.bz2 .
COPY ./scr-linux-patch.tgz .

# Unpack TORCS and patch
RUN tar xvjf torcs-1.3.4.tar.bz2 && \
    tar xvzf scr-linux-patch.tgz -C torcs-1.3.4

# Apply the patch
WORKDIR /opt/torcs-1.3.4/scr-patch
RUN sh do_patch.sh

# Fix compilation error in OpenALMusicPlayer.cpp
RUN sed -i "s/const char\* error = '\\\\0';/const char\* error = NULL;/g" /opt/torcs-1.3.4/src/libs/musicplayer/OpenALMusicPlayer.cpp

# Fix isnan compilation error in geometry.cpp
RUN sed -i "s/isnan(/std::isnan(/g" /opt/torcs-1.3.4/src/drivers/olethros/geometry.cpp

# Build TORCS with SCR patch
WORKDIR /opt/torcs-1.3.4
RUN ./configure && \
    make && \
    make install && \
    make datainstall

# Create TORCS config directory
RUN mkdir -p /root/.torcs/config/raceman

# Copy headless practice configuration
COPY ./practice.xml /root/.torcs/config/raceman/practice.xml

# Create startup script that sets up xvfb and proper environment
RUN echo '#!/bin/bash\n\
export LD_LIBRARY_PATH=/usr/local/lib/torcs/lib:$LD_LIBRARY_PATH\n\
export DISPLAY=:99\n\
Xvfb :99 -screen 0 1024x768x24 &\n\
XVFB_PID=$!\n\
cd /usr/local/share/games/torcs\n\
/bin/bash /usr/local/lib/torcs/setup_linux.sh /root/.torcs\n\
sleep 2\n\
torcs "$@"\n\
kill $XVFB_PID 2>/dev/null\n' > /usr/local/bin/torcs-headless

RUN chmod +x /usr/local/bin/torcs-headless

# Set TORCS to run in headless mode with virtual display
CMD ["torcs-headless", "-r", "/root/.torcs/config/raceman/practice.xml", "-nofuel", "-nodamage", "-nolaptime"]