FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /root/workspace/pipelines/train/LLM/sft/naive_torch

# Add sudo
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    sudo \
    build-essential \
    software-properties-common \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \

RUN sudo add-apt-repository ppa:deadsnakes/ppa -y && \
    sudo apt-get update && \
    sudo apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-pip && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

CMD ["bash"]