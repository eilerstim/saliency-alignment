FROM nvcr.io/nvidia/pytorch:25.02-py3

# Configure project dir
ARG PROJECT_DIR=/workspace
ENV PROJECT_DIR="${PROJECT_DIR}"

# Set working directory
WORKDIR ${PROJECT_DIR}

# Caches inside mounted directory
ENV HF_HOME="${PROJECT_DIR}/.hf"

ENTRYPOINT ["/bin/bash"]
CMD ["-l"]