FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH $CONDA_DIR/bin:$PATH

# Initialize conda in bash
RUN conda init bash

#######################################
# Set the working directory
RUN mkdir workspace
WORKDIR /workspace

#######################################
# Create llama.cpp environment
RUN conda create -n llama.cpp python=3.10 -y -c conda-forge

# Install torch
RUN conda run -n llama.cpp pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Copy the requirements.txt file into the container
COPY requirements_llama.cpp.txt /workspace/requirements_llama.cpp.txt

# Install the Python packages specified in requirements.txt
RUN conda run -n llama.cpp pip3 install -r requirements_llama.cpp.txt

# Install llama-cpp-python with GPU support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN conda run -n llama.cpp pip3 install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Install pytables:
RUN conda run -n llama.cpp conda install -c conda-forge pytables

#######################################
# Create transformers environment
RUN conda create -n transformers python=3.10 -y -c conda-forge

# Install torch
RUN conda run -n transformers pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Copy the requirements.txt file into the container
COPY requirements_transformers.txt /workspace/requirements_transformers.txt

# Install the Python packages specified in requirements.txt
RUN conda run -n transformers pip3 install -r requirements_transformers.txt

# Install pytables:
RUN conda run -n transformers conda install -c conda-forge pytables numpy==1.26.4

# Cleanup step after each conda/pip installation to reduce image size
RUN conda clean -afy && \
    pip cache purge

#######################################

# Set the entrypoint to use conda run
ENTRYPOINT ["conda", "run", "-n"]

# Set a default command
CMD ["bash", "-c", "echo 'Please specify an environment: llama.cpp or transformers'"]