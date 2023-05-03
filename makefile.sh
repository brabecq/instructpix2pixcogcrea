#!/bin/bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

sudo cog build -t instructpix2pixcogcrea
sudo docker run -d -p 5000:5000 --gpus all instructpix2pixcogcrea
