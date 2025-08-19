# face-blur-server
server implemented in kube to blur faces on images  
It works in two steps. First, the server-flask waits for a photo. When it receives one, it launches a network that detects the face's bbox, and then (second step) launches a job in which the image is blurred.
---
## Technologies:
- **Lang:** Python
- **Containerization & IaC:** Docker, Kubernetes
- **ML:** PyTorch, NumPy, PIL, matplotlib
- **Libs:** Flask, requests
---
## Requirements
- **env:** docker, minikube
- **dependencies:** requirements.txt per image
- **acceleration:** (optional) CUDA linux
---
## Installation
- build images
- run k8s
---

