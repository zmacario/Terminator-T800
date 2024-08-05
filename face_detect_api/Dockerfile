FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get install -y python3 python3-pip libopenblas-base libopenmpi-dev libomp-dev libcudnn9*
RUN apt-get install -y cuda-toolkit-12-4
WORKDIR /usr/src/app
COPY ./ /usr/src/app/
RUN pip3 install -r requirements.txt
EXPOSE 8002
CMD ["python3", "main.py"]