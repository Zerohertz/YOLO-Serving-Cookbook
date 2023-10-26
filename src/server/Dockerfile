FROM nvcr.io/nvidia/tritonserver:23.06-py3

COPY ./requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./ /models/