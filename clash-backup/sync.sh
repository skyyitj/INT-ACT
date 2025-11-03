#!/bin/bash

# Variables
FILE_PATH="/home/liuchi/clash/README.txt"
REMOTE_USER="liuchi"
REMOTE_IP="10.1.114.77"
REMOTE_PATH="/home/liuchi/clash/README.txt"

# Sync file to remote machine
scp $FILE_PATH $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH
