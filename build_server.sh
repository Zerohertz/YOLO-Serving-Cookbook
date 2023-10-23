docker pull nvcr.io/nvidia/tritonserver:23.06-py3
docker run -itd -e NVIDIA_VISIBLE_DEVICES=0 \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v ${PWD}/server:/models \
--name triton-inference-server \
nvcr.io/nvidia/tritonserver:23.06-py3 \
tritonserver --model-repository=/models && \
docker logs -f triton-inference-server

# pip install tritonclient
# I1023 07:24:30.623635 1 grpc_server.cc:2445] Started GRPCInferenceService at 0.0.0.0:8001
# I1023 07:24:30.623837 1 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
# I1023 07:24:30.665074 1 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002