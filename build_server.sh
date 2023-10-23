docker pull nvcr.io/nvidia/tritonserver:23.06-py3
docker run -itd -e NVIDIA_VISIBLE_DEVICES=0 \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v ${PWD}/server:/models \
--name triton-inference-server \
nvcr.io/nvidia/tritonserver:23.06-py3 \
tritonserver --model-repository=/models && \
docker logs -f triton-inference-server
