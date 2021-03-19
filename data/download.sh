#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color

# Download and decompress datasets

if [ ! -d "speech_commands/train/" ]; 
then
    echo "Downloading Speech Commands dataset(about 2.3GB)..."
    wget -O speech_commands/speech_command_preprocessed.tar.gz https://www.dropbox.com/s/iyv9nomth713qbc/speech_command_preprocessed.tar.gz?dl=0

    echo "Dataset downloaded, now decompressing..." 
    tar -xf speech_commands/speech_command_preprocessed.tar.gz

    echo "Removing compressed file..."
    rm -f speech_commands/speech_command_preprocessed.tar.gz

    echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
else
    echo -e "${RED}Speech Commands dataset already exists under speech_commands/!"
fi

if [ ! -d "stackoverflow/train/" ]; 
then
    echo "Downloading Stack Overflow dataset(about 3.6GB)..."
    wget -O stackoverflow/stackoverflow_preprocessed.tar.gz https://www.dropbox.com/s/8dh6b33t3n1d16i/stackoverflow_preprocessed.tar.gz?dl=0

    echo "Dataset downloaded, now decompressing..." 
    tar -xf stackoverflow/stackoverflow_preprocessed.tar.gz

    echo "Removing compressed file..."
    rm -f stackoverflow/stackoverflow_preprocessed.tar.gz

    echo -e "${GREEN}Stack Overflow dataset downloaded!${NC}"
else
    echo -e "${RED}Stack Overflow dataset already exists under stackoverflow/!"
fi


if [ ! -d "open_images/train/" ]; 
then
    echo "Downloading Open Images dataset(about 66GB)..."
    wget -O 

    echo "Dataset downloaded, now decompressing..." 
    tar -xf 

    echo "Removing compressed file..."
    rm -f 

    echo -e "${GREEN}Open Images dataset downloaded!${NC}"
else
    echo -e "${RED}Open Images dataset already exists under open_images/!"
fi

if [ ! -d "reddit/train/" ]; 
then
    echo "Downloading Reddit dataset(about 3.8GB)..."
    wget -O reddit/reddit_preprocessed.tar.gz https://www.dropbox.com/s/1mdu98bohm35uft/reddit_preprocessed.tar.gz?dl=0

    echo "Dataset downloaded, now decompressing..." 
    tar -xf reddit/reddit_preprocessed.tar.gz

    echo "Removing compressed file..."
    rm -f reddit/reddit_preprocessed.tar.gz

    echo -e "${GREEN}Reddit dataset downloaded!${NC}"
else
    echo -e "${RED}Open Images dataset already exists under reddit/!"
fi

if [ ! -d "misc/" ]; 
then
    echo "Downloading preprocessed dataset stats(about 100MB)..."
    mkdir stats
    wget -O misc/openimg_distr.pkl https://www.dropbox.com/s/otnxfyfm6dqug9g/openimg_size.pkl?dl=0
    wget -O misc/reddit_distr.pkl https://www.dropbox.com/s/c08tnwv479u8h41/reddit_size.pkl?dl=0
    wget -O misc/speech_distr.pkl https://www.dropbox.com/s/rjce3wqf4umjcae/speech_size.pkl?dl=0
    wget -O misc/stackoverflow_distr.pkl https://www.dropbox.com/s/0yujx9hgybyb4nl/stackoverflow_size.pkl?dl=0

    echo -e "${GREEN}Dataset stats downloaded!${NC}"
else
    echo -e "${RED}Dataset stats already exists under stats/!"
fi