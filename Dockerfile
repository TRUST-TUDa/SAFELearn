# Use the official PyTorch image with GPU support as the base image
FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgmp3-dev \
    libssl-dev \
    libboost-all-dev \
    curl \
    neovim

WORKDIR /
# Download ABY and SAFELearn
RUN git clone https://github.com/encryptogroup/ABY.git && git clone https://github.com/mavonarx/SAFELearn.git

# Compile ABY
WORKDIR /ABY
RUN mkdir -p build && cd build && cmake .. && make && make install

# Adjust variables in CMakeLists.txt and build SAFELearn
WORKDIR /SAFELearn
RUN sed -i 's|ABSOLUTE_PATH_TO_ABY|/ABY/build|g' ./CMakeLists.txt \
    && sed -i 's|/include|/extern/ENCRYPTO_utils/include|g' ./CMakeLists.txt \
    && mkdir -p build && mkdir -p model && cd build && cmake .. && make

# Get pip and use it for installing Python modules
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py
RUN python3 -m pip install numpy && python3 -m pip install scikit-learn
RUN python3 -m pip install scikit-learn pandas torcheval

# Allow colors in bash
RUN echo 'PS1="\[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]"' >> /root/.bashrc \
    && echo "alias ls=\"ls --color=auto\"" >> /root/.bashrc \
    && echo "alias python=\"python3\"" >> /root/.bashrc

CMD ["/bin/bash"]

