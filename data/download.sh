#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="."

# Download and decompress datasets

Help()
{
   # Display Help
   echo "We provide four datasets (open_images, reddit, stackoverflow, and speech)"
   echo "to evalute the performance of Kuiper"
   echo 
   echo "Syntax: ./download.sh [-g|h|v|V]"
   echo "options:"
   echo "-h     Print this Help."
   echo "-A     Download all datasets (about 77GB)"
   echo "-r     Download Reddit dataset (about 3.8GB)"
   echo "-t     Download Stack Overflow dataset (about 3.6GB)"
   echo "-p     Download Speech Commands dataset (about 2.3GB)"
   echo "-o     Download Open Images dataset (about 66GB)"
   echo
}

speech()
{
    if [ ! -d "speech_commands/train/" ]; 
    then
        echo "Downloading Speech Commands dataset(about 2.3GB)..."
        # wget -O speech_commands/speech_command_preprocessed.tar.gz https://www.dropbox.com/s/iyv9nomth713qbc/speech_command_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        # tar -xf speech_commands/speech_command_preprocessed.tar.gz

        echo "Removing compressed file..."
        # rm -f speech_commands/speech_command_preprocessed.tar.gz

        echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
    else
        echo -e "${RED}Speech Commands dataset already exists under speech_commands/!"
fi
}

stackoverflow()
{
    if [ ! -d "stackoverflow/train/" ]; 
    then
        echo "Downloading Stack Overflow dataset(about 3.6GB)..."
        # wget -O stackoverflow/stackoverflow_preprocessed.tar.gz https://www.dropbox.com/s/8dh6b33t3n1d16i/stackoverflow_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        # tar -xf stackoverflow/stackoverflow_preprocessed.tar.gz

        echo "Removing compressed file..."
        # rm -f stackoverflow/stackoverflow_preprocessed.tar.gz

        echo -e "${GREEN}Stack Overflow dataset downloaded!${NC}"
    else
        echo -e "${RED}Stack Overflow dataset already exists under stackoverflow/!"
fi
}

open_images() 
{
    if [ ! -d "open_images/train/" ]; 
    then
        echo "Downloading Open Images dataset(about 66GB)..."
        # wget -O 

        echo "Dataset downloaded, now decompressing..." 
        # tar -xf 

        echo "Removing compressed file..."
        # rm -f 

        echo -e "${GREEN}Open Images dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under open_images/!"
fi
}

reddit()
{
    if [ ! -d "reddit/train/" ]; 
    then
        echo "Downloading Reddit dataset(about 3.8GB)..."
        # wget -O reddit/reddit_preprocessed.tar.gz https://www.dropbox.com/s/1mdu98bohm35uft/reddit_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        # tar -xf reddit/reddit_preprocessed.tar.gz

        echo "Removing compressed file..."
        # rm -f reddit/reddit_preprocessed.tar.gz

        echo -e "${GREEN}Reddit dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under reddit/!"
    fi
}

stats()
{
    if [ ! -d "stats/" ]; 
    then
        echo "Downloading preprocessed dataset stats(about 100MB)..."
        # mkdir stats
        # wget -O stats/openimg_distr.pkl https://www.dropbox.com/s/otnxfyfm6dqug9g/openimg_size.pkl?dl=0
        # wget -O stats/reddit_distr.pkl https://www.dropbox.com/s/c08tnwv479u8h41/reddit_size.pkl?dl=0
        # wget -O stats/speech_distr.pkl https://www.dropbox.com/s/rjce3wqf4umjcae/speech_size.pkl?dl=0
        # wget -O stats/stackoverflow_distr.pkl https://www.dropbox.com/s/0yujx9hgybyb4nl/stackoverflow_size.pkl?dl=0

        echo -e "${GREEN}Dataset stats downloaded!${NC}"
    else
        echo -e "${RED}Dataset stats already exists under stats/!"
    fi
}


while getopts ":hAarotp" option; do
   case $option in
      h ) # display Help
         Help
         exit;;
      A )
         speech
         open_images
         reddit
         stackoverflow
         stats
         exit;;
      a )
         stats   
         ;;
      o )
         open_images   
         ;;  
      r )
         reddit   
         ;;          
      t )
         stackoverflow   
         ;;  
      p )
         speech   
         ;;                    
      \? ) 
         echo -e "${RED}Usage: cmd [-h] [-A] [-r] [-o] [-t] [-p]${NC}"
         exit 1;;
   esac
done

if [ $OPTIND -eq 1 ]; then 
    echo -e "${RED}Usage: cmd [-h] [-A] [-r] [-o] [-t] [-p]${NC}"; 
fi