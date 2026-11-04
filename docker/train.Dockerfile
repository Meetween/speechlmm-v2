ARG BASE_IMAGE_NAME=meetween/meetween_base_cuda12-8
ARG BASE_IMAGE_VERSION=0.1.2
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_VERSION}

ENV ROOT=/workspace
ENV CODE_DIR=$ROOT/speechlmm-v2

WORKDIR $CODE_DIR

# install ffmpeg
RUN apt update && apt install -y ffmpeg

# copy source code of speechlmm-v2 (git-clone from outside)
COPY . $CODE_DIR

# store the current commit hash into a file
RUN cd $CODE_DIR/; git log -n 1 > COMMIT.log ; cat ./COMMIT.log

# store the current submodules into a file
RUN cd $CODE_DIR/ ; git submodule > SUBMODULE.log ; cat ./SUBMODULE.log

# create virtual env
RUN uv venv $ROOT/speechlmm-v2_env --python=3.12

ENV VIRTUAL_ENV=$ROOT/speechlmm-v2_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install speechlmm-v2
RUN cd $CODE_DIR/ ; uv pip install -e ".[deepspeed]"

# install wandb
RUN uv pip install wandb

# install tensorboardX
RUN uv pip install tensorboardX

CMD ["bash"]
