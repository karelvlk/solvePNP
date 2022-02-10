#!/bin/bash

while getopts "t:v" option; do
    case $option in
        t ) type=${OPTARG};;
        v ) version=${OPTARG};;
    esac
done

function runTest() {

    volume="$(pwd):/app"

    if [[ $OSTYPE == "msys" ]]; then
        volume="/$volume"
    fi

    docker build -f ./Dockerfile  . -t naki-jeseniky-solvepnp
    docker build -f ./Dockerfile-test  . -t naki-jeseniky-solvepnp-test
    docker run -v $volume -d --name service --rm naki-jeseniky-solvepnp
    sleep 5
    docker run -v $volume --link service:service --rm naki-jeseniky-solvepnp-test

    docker logs service
    docker stop service
}

case $type in
    test)
        runTest
        ;;
esac

