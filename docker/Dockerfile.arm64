# This docker container is used for testing pygeos in ARM64 emulation mode.
# To build it:
# docker build . -f ./docker/Dockerfile.arm64 -t pygeos/arm64
# Then run the pygeos test suite:
# docker run --rm pygeos/arm64:latest python3 -m pytest -vv

FROM --platform=linux/arm64/v8 arm64v8/ubuntu:20.04

RUN apt-get update && apt-get install -y build-essential libgeos-dev python3-dev python3-pip --no-install-recommends

RUN pip3 install numpy Cython pytest

COPY . /code

WORKDIR /code

RUN python3 setup.py build_ext --inplace
