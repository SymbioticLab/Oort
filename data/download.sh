#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="./data"

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
    if [ ! -d "${DIR}/speech_commands/train/" ]; 
    then
        echo "Downloading Speech Commands dataset(about 2.3GB)..."
        wget -O ${DIR}/speech_commands/speech_command_preprocessed.tar.gz https://www.dropbox.com/s/iyv9nomth713qbc/speech_command_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/speech_commands/speech_command_preprocessed.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/speech_commands/speech_command_preprocessed.tar.gz

        echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
    else
        echo -e "${RED}Speech Commands dataset already exists under ${DIR}/speech_commands/!"
fi
}

stackoverflow()
{
    if [ ! -d "${DIR}/stackoverflow/train/" ]; 
    then
        echo "Downloading Stack Overflow dataset(about 3.6GB)..."
        wget -O ${DIR}/stackoverflow/stackoverflow_preprocessed.tar.gz https://www.dropbox.com/s/8dh6b33t3n1d16i/stackoverflow_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/stackoverflow/stackoverflow_preprocessed.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/stackoverflow/stackoverflow_preprocessed.tar.gz

        echo -e "${GREEN}Stack Overflow dataset downloaded!${NC}"
    else
        echo -e "${RED}Stack Overflow dataset already exists under ${DIR}/stackoverflow/!"
fi
}

open_images() 
{
    if [ ! -d "${DIR}/open_images/train/" ]; 
    then
        echo "Downloading Open Images dataset(about 66GB)..."
        wget -O ${DIR}/open_images_v5_preprocessed.tar.gz https://www.dropbox.com/s/hx2adz4761rqug2/open_images_v5_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/open_images_v5_preprocessed.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/open_images_v5_preprocessed.tar.gz

        echo -e "${GREEN}Open Images dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under ${DIR}/open_images/!"
fi
}

reddit()
{
    if [ ! -d "${DIR}/reddit/train/" ]; 
    then
        echo "Downloading Reddit dataset(about 3.8GB)..."
        wget -O ${DIR}/reddit/reddit_preprocessed.tar.gz https://www.dropbox.com/s/1mdu98bohm35uft/reddit_preprocessed.tar.gz?dl=0

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/reddit/reddit_preprocessed.tar.gz

        echo "Removing compressed file..."
        rm -f ${DIR}/reddit/reddit_preprocessed.tar.gz

        echo -e "${GREEN}Reddit dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under ${DIR}/reddit/!"
    fi
}

stats()
{
    if [ ! -d "${DIR}/misc/" ]; 
    then
        echo "Downloading preprocessed dataset stats used to reproduce figures..."
        mkdir ${DIR}/misc
        wget -O ${DIR}/misc/speech_samples_f16.pkl https://www.dropbox.com/s/zzkuy48xl68fzhs/speech_samples_f16.pkl?dl=0
        wget -O ${DIR}/misc/openimg_distr.pkl https://www.dropbox.com/s/pb2aomq7z9nn6vg/openimg_distr.pkl?dl=0

        echo -e "${GREEN}Dataset stats downloaded!${NC}"
    else
        echo -e "${RED}Dataset stats already exists under ${DIR}/stats/!"
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
        #  reddit
        #  stackoverflow
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