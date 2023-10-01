# Base imaage 
FROM debian:bullseye 

ENV DEBIAN_FRONTEND=noninteractive

# Download dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgmp3-dev \
    libssl-dev \
    libboost-all-dev \
    zsh

# Download ABY
RUN git clone https://github.com/encryptogroup/ABY.git

# Download Safelearn
RUN git clone https://github.com/TRUST-TUDa/SAFELearn.git

# Compile ABY
WORKDIR /ABY
RUN mkdir build && cd build && cmake .. && make && make install 

# Adjust variables in CMakeLists.txt and build Safelearn
WORKDIR /SAFELearn
RUN sed -i 's|ABSOLUTE_PATH_TO_ABY|/ABY/build|g' ./CMakeLists.txt
RUN sed -i 's|/include|/extern/ENCRYPTO_utils/include|g' ./CMakeLists.txt
RUN mkdir build && cd build && cmake .. && make

# allow colors in bash
RUN echo 'PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc
RUN echo "alias ls=\"ls --color=auto\"" >> /root/.bashrc
CMD ["/bin/bash"]
