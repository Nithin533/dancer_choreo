# Dance Choreography AI

## Your Machine (FastAPI pipeline)
pip install -r requirements.txt
uvicorn api:app --reload

## Friend's Machine (GPU Training)

### 1. Install Python 3.9 from python.org

### 2. Install CUDA 11.8
https://developer.nvidia.com/cuda-downloads

### 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

### 4. Install requirements
pip install -r requirements.txt

### 5. Copy data folders into project root
motions/
wav/
dataset/

### 6. Train
python train.py

### 7. After training — copy this back
checkpoints/best_model.pt