#!/bin/bash

usage() {
    echo --tag [-t]: tag for docker image
    echo --build [-b]: if present, build the Dockerfile
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

TAG=
DOCKERPATH=
BUILD=false
while [ "$1" != "" ]; do
    case $1 in
        -t | --tag)
            shift
            TAG=$1
            ;;
        -p | --dockerfile-path)
            shift
            DOCKERPATH=$1
            ;;
        -b | --build)
            BUILD=true
            ;;
        *)
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$TAG" == "" ]; then
    echo "error: tag must not be blank"
    exit 1
fi

if $BUILD; then
    if [ "$DOCKERPATH" == "" ]; then
        echo "error: Dockerfile path must not be blank"
        exit 1
    fi
    nvidia-docker build $DOCKERPATH -t $TAG
fi

nvidia-docker rmi $(nvidia-docker images -q -f "label=taost" -f "dangling=true")

nvidia-docker run -p 127.0.0.1:8800:8800 --rm --name $TAG $TAG:latest

