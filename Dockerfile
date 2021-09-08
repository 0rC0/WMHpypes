# syntax=docker/dockerfile:1
# Use Ubuntu 20.04
FROM ubuntu:20.04

# Install Ubuntu packages
RUN apt-get update && \
    apt-get install --yes \
                    curl \
                    git

# Installing and SetUp MiniConda latest
RUN curl -sSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/miniconda/bin:$PATH"

# Install conda environment
RUN conda install --yes  -c conda-forge tensorflow \
                                        scipy \
                                        keras \
                                        numpy \
                                        nibabel \
                                        nipype \
                                        simpleitk && \
        chmod -R a+rX /opt/miniconda; sync && \
        chmod +x /opt/miniconda/bin/*; sync && \
        conda build purge-all; sync && \
        conda clean -tipsy && sync

# NiPype environmental variables
ENV MKL_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Create home directory
RUN useradd -m -s /bin/bash -G users wmhpypes
ENV HOME="/home/wmhpypes"
# Install WMHpypes
RUN git clone https://github.com/0rC0/WMHpypes.git $HOME/wmhpypes && \
    cd $HOME/wmhpypes && \
    pip install .
ENV PATH="$HOME/wmhpypes/scripts":$PATH