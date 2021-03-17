#!/bin/bash

# Download and decompress datasets

if [ ! -d "speech_commands/train/" ]; then
    echo "Downloading Speech Commands dataset(about 2.3GB)..."
    wget -O speech_commands/speech_command_preprocessed.tar.gz https://www.dropbox.com/s/iyv9nomth713qbc/speech_command_preprocessed.tar.gz?dl=0

    echo "Dataset downloaded, now decompressing..." 
    tar -xf speech_commands/speech_command_preprocessed.tar.gz

    echo "Removing compressed file..."
    rm -f speech_commands/speech_command_preprocessed.tar.gz

    echo "Speech Commands dataset downloaded..."
fi

if [ ! -d "stackoverflow/train/" ]; then
    echo "Downloading stackoverflow dataset(about 3.6GB)..."
    wget -O stackoverflow/stackoverflow_preprocessed.tar.gz https://www.dropbox.com/s/8dh6b33t3n1d16i/stackoverflow_preprocessed.tar.gz?dl=0

    echo "Dataset downloaded, now decompressing..." 
    tar -xf stackoverflow/stackoverflow_preprocessed.tar.gz

    echo "Removing compressed file..."
    rm -f stackoverflow/stackoverflow_preprocessed.tar.gz

    echo "stackoverflow commands dataset downloaded..."
fi

if [ ! -d "open_images/train/" ]; then
    echo "Downloading Open Images dataset(about xxGB)..."
    wget -O 

    echo "Dataset downloaded, now decompressing..." 
    tar -xf 

    echo "Removing compressed file..."
    rm -f 

    echo "Open Images commands dataset downloaded..."
fi

if [ ! -d "reddit/train/" ]; then
    echo "Downloading Reddit dataset(about 3.8GB)..."
    wget -O reddit/reddit_preprocessed.tar.gz https://www.dropbox.com/s/1mdu98bohm35uft/reddit_preprocessed.tar.gz?dl=0

    echo "Dataset downloaded, now decompressing..." 
    tar -xf reddit/reddit_preprocessed.tar.gz

    echo "Removing compressed file..."
    rm -f reddit/reddit_preprocessed.tar.gz

    echo "Reddit commands dataset downloaded..."
fi