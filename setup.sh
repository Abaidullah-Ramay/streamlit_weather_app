#!/bin/bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libasound-dev ffmpeg

# Fix library path configuration
echo "/usr/lib/x86_64-linux-gnu" | sudo tee /etc/ld.so.conf.d/portaudio.conf
sudo ldconfig