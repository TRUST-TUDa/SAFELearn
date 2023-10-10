#!/bin/bash

# build the image
docker build -t maettu102/safelearn:latest . --no-cache

# push it
docker push maettu102/safelearn:latest
