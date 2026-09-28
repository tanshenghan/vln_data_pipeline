# Context is produced by configs/prepare_cloud_image.py, NOT the repository root.
# R2R/RxR image. Isaac/LightNav/VLM dependencies need a separately validated image.
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ffmpeg ca-certificates libegl1 libgl1 libglvnd0 \
    libglx0 libopengl0 libgomp1 libglib2.0-0 libx11-6 libxcursor1 libxi6 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*
ADD habitat.tar.gz /opt/habitat/
ADD python311.tar.gz /opt/python311/
RUN python3.10 -m venv /opt/platform && /opt/python311/bin/python3.11 -m venv /opt/lerobot
ADD platform-site.tar.gz /opt/platform/lib/python3.10/site-packages/
ADD lerobot-site.tar.gz /opt/lerobot/lib/python3.11/site-packages/
COPY code/ /opt/VLN_Data_Pipeline/
COPY build-manifest.json /opt/VLN_Data_Pipeline/cloud-image-manifest.json
WORKDIR /opt/VLN_Data_Pipeline
ENV PATH=/opt/platform/bin:$PATH \
    PYTHONPATH=/opt/VLN_Data_Pipeline:/opt/VLN_Data_Pipeline/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLN_SOURCE_ROOT=/mnt/source \
    VLN_RESOURCE_ROOT=/mnt/cloud \
    VLN_HABITAT_PYTHON=/opt/habitat/bin/python \
    VLN_LEROBOT_PYTHON=/opt/lerobot/bin/python \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# Import checks require no GPU, source data or cloud credentials.
RUN /opt/habitat/bin/python -c "import habitat_sim, numpy, PIL; print(habitat_sim.__version__)" \
    && /opt/lerobot/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; import torch; print(torch.__version__)" \
    && VLN_DATABASE_URL=sqlite:// /opt/platform/bin/python -c "from workers.tasks import collect_gt; from workers.cloud_entry import validate_environment" \
    && /opt/platform/bin/python -m workers.cloud_entry --help
# Atomic control/result files are private to the volume owner. Match the
# storage writer's numeric UID/GID (the remote login for an idmap=user bridge).
ARG RUN_UID=1000
ARG RUN_GID=1000
RUN groupadd --gid ${RUN_GID} vln \
    && useradd --uid ${RUN_UID} --gid ${RUN_GID} --create-home --home-dir /home/vln vln
ENV HOME=/home/vln
USER ${RUN_UID}:${RUN_GID}
CMD ["/opt/platform/bin/python", "-m", "workers.cloud_entry", "--help"]
