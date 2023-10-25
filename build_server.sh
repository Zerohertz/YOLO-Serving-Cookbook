kubectl create namespace yolo
kubectl apply -f triton-inference-server.yaml -n yolo
kubectl apply -f fastapi.yaml -n yolo
kubectl apply -f ingress.yaml -n yolo