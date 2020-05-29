#! /bin/sh

D_USER=$(whoami)
D_UID=$(id -u)
D_GID=$(id -g)

# Build the image
nvidia-docker build --build-arg=CONTAINER_USER=${D_USER} --build-arg=CONTAINER_UID=${D_UID}  -t args_snippet_gen .

# Run the container
nvidia-docker run -u ${D_UID}:${D_GID} --name args-snippet-generation -p 5000:5000 -it args_snippet_gen:latest bash    .

