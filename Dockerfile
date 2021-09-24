FROM python:3.9-slim-buster

RUN apt-get -y update && apt-get install -y build-essential wget

ENV OPENMPI_VERSION 4.0.5
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget -q https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar -xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN apt-get -y update && apt-get install -y ssh zlib1g-dev libjpeg-dev swig

WORKDIR /nes

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

RUN python -m pip install "box2d-py==2.3.8"
RUN python -m pip install "gym==0.18.0"
RUN python -m pip install "mpi4py==3.0.3"

COPY nes nes
COPY example.py example.py
