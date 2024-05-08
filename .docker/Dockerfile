FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_frontend noninteractive
ENV TZ=Etc/UTC
ENV LANG en_US.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y \
    sudo \
    git \
    unzip \
    tar \
    nano \
    vim \
    gedit \
    make \
    cmake \
    curl \
    python3-pip \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/*


RUN adduser --disabled-password --gecos '' captain \
    && adduser captain sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER captain
WORKDIR /home/captain/
RUN chmod a+rwx /home/captain/


RUN sudo apt-get update && sudo apt install -y software-properties-common && \
    sudo add-apt-repository universe && \
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    sudo apt update && \
    sudo apt install -y ros-humble-desktop ros-dev-tools && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    sudo rm -rf /var/lib/apt/lists/*


RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
