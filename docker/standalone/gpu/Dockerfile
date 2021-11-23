FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer caffe-maint@googlegroups.com
RUN apt-get update && apt-get install -y --no-install-recommends 
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0

RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev libatlas-base-dev git

RUN  apt-get -y install --no-install-recommends libboost-all-dev

RUN git clone --depth 1 https://github.com/RatkoFri/minsoo_caffe.git . && \
    pip install --upgrade "pip < 20"  && \
    cd python && for req in $(cat requirements.txt) pydot; do python -m pip install $req; done && cd .. 

RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl

RUN whereis libcudnn.so

RUN apt-get update && apt-get  -y install doxygen texlive ruby-dev 
RUN apt-get update && gem install jekyll execjs therubyracer

RUN apt-get purge -y --auto-remove cmake

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.3-Linux-x86_64.sh /cmake-3.12.3-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.3-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

RUN mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCUDA_ARCH_NAME=Volta .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace