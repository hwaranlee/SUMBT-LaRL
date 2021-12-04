FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

LABEL maintainer="Hwaran Lee <hwaran.lee@sktbrain.com>"

ARG PYTHON_VERSION=3.6

# Install some basic utilities
RUN apt-get update --fix-missing && apt-get install -y \
	build-essential \
	curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
	vim \
	cmake \
	tmux \
	openssh-server \
    xvfb \
	gcc \
	libc-dev \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender1 \
	wget \
	libevent-dev \
	build-essential \
	openjdk-8-jdk && \
	rm -rf /var/lib/apt/lists/*

# Setup locale
RUN	apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
RUN locale-gen en_US.UTF-8

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash
	
ENV PATH=$CONDA_DIR/bin:$PATH

# Install conda as jovyan and check the md5 sum provided on the download site
ENV MINICONDA_VERSION=4.5.11
RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "e1045ee415162f944b6aebfe560b8fee *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda install --quiet --yes conda="${MINICONDA_VERSION%.*}.*" && \
    $CONDA_DIR/bin/conda update --all --quiet --yes && \
    conda clean -tipsy

# Create a Python 3.6 environment
RUN $CONDA_DIR/bin/conda install conda-build \
 && $CONDA_DIR/bin/conda create -y --name py36 python=3.6.5 \
 && $CONDA_DIR/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install pytorch CUDA 10.1 -specific steps
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

# Install tensorflow
RUN pip install tensorflow-gpu

# Install nltk
RUN pip install --upgrade pip
RUN pip install -U nltk

# Install ConvLab
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]

# Install BERT
RUN pip install pytorch-pretrained-bert==0.6.1
RUN pip install tensorboardX
RUN pip install tqdm

WORKDIR /root
