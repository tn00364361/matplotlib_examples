#!/usr/bin/bash

PASSWD_FILE=$(mktemp)
GROUP_FILE=$(mktemp)
echo $(getent passwd $(id -un)) > $PASSWD_FILE
echo $(getent group $(id -un)) > $GROUP_FILE

xhost +local:$USER

docker run -it --rm \
    --network host \
    --shm-size 4g \
    --device /dev/dri \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME \
    -e USER \
    -e XDG_RUNTIME_DIR=/run/user/$(id -u) \
    -u $(id -u):$(id -g) \
    -v /run/user/$(id -u):/run/user/$(id -u) \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PASSWD_FILE:/etc/passwd:ro \
    -v $GROUP_FILE:/etc/group:ro \
    -v $(pwd)/docker/home:$HOME \
    -v $(pwd):/mpl_examples \
    -w /mpl_examples \
    mpl_examples

xhost -local:$USER
