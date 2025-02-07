# Model Conversion and Profiling Project

## Overview

This project implements a pipeline for converting and profiling deep learning models across PyTorch, ONNX, and TensorRT formats. It includes model training, conversion, and comprehensive performance profiling.

## Project Structure
```
├── assets/
│   ├── model.onnx
│   ├── model.trt
│   └── pt_model.pth
├── Dataset/
│   └── melanoma_cancer_dataset/
│       ├── train/
│       │   ├── benign/
│       │   └── malignant/
│       └── test/
│           ├── benign/
│           └── malignant/
├── ModelWrappers/
│   ├── __init__.py
│   ├── Onnx_Wrapper.py
│   ├── Pytorch_Wrapper.py
│   └── TensorRT_Wrapper.py
├── Profiling/
│   ├── __init__.py
│   └── profiler.py
├── main.py
├── requirements.txt
└── model_profiling_results.csv

```

## Requirements
```
numpy
Pillow
torch
torchvision
pycuda
tensorrt
pynvml

```

## Usage

1. Training PyTorch Model

```python 
pt_model = Classifier(
    train_data_path='Dataset/melanoma_cancer_dataset/train',
    test_data_path='Dataset/melanoma_cancer_dataset/test',
    model_name='resnet18',
    num_classes=1,
    batch_size=32,
    lr=0.0001,
    num_epochs=20
)
pt_model.train()
pt_model.save_model('assets/pt_model.pth')
```
2. Converting to ONNX

```python
onnx_model = OnnxWrapper()
onnx_model.Torch2Onnx(
    pt_model_path='assets/pt_model.pth',
    input_size=(1, 3, 224, 224),
    output_onnx_path='assets/model.onnx'
)
```
3. Converting to TensorRT

```python
trt_model = TensorRTWrapper(model_path='assets/model.onnx', quantize="fp16")
trt_model.build_engine()
```

# Model Profiling Results

| Model Type | Accuracy (%) | Avg Inference Time (s) | Avg Memory Usage (MB) |
|------------|--------------|-------------------------|------------------------|
| PyTorch    | 89.2         | 0.0116                 | 0.5141                |
| ONNX       | 89.2         | 0.0151                 | 0.0052                |
| TensorRT   | 89.5         | 0.0022                 | 0.0007                |

## Key Observations:
- All models maintain similar accuracy (~89%).
- TensorRT achieves the fastest inference time (2.2ms).
- TensorRT has the lowest memory usage (0.0007MB).
- Model conversion preserves accuracy while improving performance.

## Components

### Model Wrappers
- **Pytorch_Wrapper.py**: Handles PyTorch model training and inference.
- **Onnx_Wrapper.py**: Manages ONNX model conversion and inference.
- **TensorRT_Wrapper.py**: Handles TensorRT model conversion and inference.

### Profiling
- **profiler.py**: Contains the `ModelProfiler` class for benchmarking models.

### Main Script
- **main.py**: Orchestrates the entire pipeline from training to profiling.

## License
This project is licensed under the MIT License.

