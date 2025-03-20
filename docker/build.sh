#!/usr/bin/bash

docker build \
    -t mpl_examples \
    -f ./docker/Dockerfile \
    .
