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
    curl \
    neovim

# Download ABY and safelearn
RUN git clone https://github.com/encryptogroup/ABY.git && git clone https://github.com/mavonarx/SAFELearn.git 

# Compile ABY
WORKDIR /ABY
RUN mkdir build && cd build && cmake .. && make && make install 

# Adjust variables in CMakeLists.txt and build Safelearn
WORKDIR /SAFELearn
RUN sed -i 's|ABSOLUTE_PATH_TO_ABY|/ABY/build|g' ./CMakeLists.txt \
&& sed -i 's|/include|/extern/ENCRYPTO_utils/include|g' ./CMakeLists.txt \
&& mkdir build && mkdir model && cd build && cmake .. && make

# get pip and use it for installing modules
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py 
RUN python3 -m pip install filelock
RUN python3 -m pip install sympy
RUN python3 -m pip install numpy
RUN python3 -m pip install networkx
RUN python3 -m pip install jinja2
RUN python3 -m pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html 


# allow colors in bash
RUN echo 'PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc \
&& echo "alias ls=\"ls --color=auto\"" >> /root/.bashrc

CMD ["/bin/bash"]
