import os
import time
import torch
import numpy as np
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import psutil
import pandas as pd
from contextlib import contextmanager
import timm
import torch.nn as nn
from typing import Dict, List, Tuple

class ModelProfiler:
    def __init__(self, test_data_path: str, batch_size:None):
        """
        Initialize the model profiler with test data path and batch size.
        """
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_dataset = datasets.ImageFolder(test_data_path, self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
    @contextmanager
    def timer_context(self):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        yield
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.current_time = end_time - start_time
        self.current_memory = end_memory - start_memory

    def profile_pytorch_model(self, model_path: str) -> Dict:
        """Profile PyTorch model"""
        # Load model
        model = timm.create_model('resnet18', pretrained=False, num_classes=1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        correct = 0
        total = 0
        total_time = 0
        total_memory = 0
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with self.timer_context():
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                total_time += self.current_time
                total_memory += self.current_memory
                correct += (preds == labels.unsqueeze(1)).sum().item()
                total += labels.size(0)
                
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        avg_inference_time = total_time / len(self.test_loader)
        avg_memory_usage = abs(total_memory / len(self.test_loader))

        return {
            'model_type': 'PyTorch',
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_memory_usage': avg_memory_usage,
            'predictions': predictions,
            'actual_labels': actual_labels
        }

    def profile_onnx_model(self, model_path: str) -> Dict:
        """Profile ONNX model"""
        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        correct = 0
        total = 0
        total_time = 0
        total_memory = 0
        predictions = []
        actual_labels = []

        for inputs, labels in self.test_loader:
            inputs = inputs.numpy()
            
            with self.timer_context():
                outputs = session.run(None, {input_name: inputs})
                preds = (torch.sigmoid(torch.tensor(outputs[0])) > 0.5).float()

            total_time += self.current_time
            total_memory += self.current_memory
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
            
            predictions.extend(preds.numpy())
            actual_labels.extend(labels.numpy())

        accuracy = 100 * correct / total
        avg_inference_time = total_time / len(self.test_loader)
        avg_memory_usage = abs(total_memory / len(self.test_loader))

        return {
            'model_type': 'ONNX',
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_memory_usage': avg_memory_usage,
            'predictions': predictions,
            'actual_labels': actual_labels
        }

    def profile_tensorrt_model(self, engine_path: str) -> Dict:
        """Profile TensorRT model"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        
        correct = 0
        total = 0
        total_time = 0
        total_memory = 0
        predictions = []
        actual_labels = []

        for inputs, labels in self.test_loader:
            input_batch = inputs.numpy()
            
            # Allocate device memory
            d_input = cuda.mem_alloc(input_batch.nbytes)
            output = np.empty((self.batch_size, 1), dtype=np.float32)
            d_output = cuda.mem_alloc(output.nbytes)
            bindings = [int(d_input), int(d_output)]

            with self.timer_context():
                # Transfer input data to device
                cuda.memcpy_htod(d_input, input_batch)
                # Execute model
                context.execute_v2(bindings=bindings)
                # Transfer predictions back
                cuda.memcpy_dtoh(output, d_output)
                preds = (torch.sigmoid(torch.tensor(output)) > 0.5).float()

            total_time += self.current_time
            total_memory += self.current_memory
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
            
            predictions.extend(preds.numpy())
            actual_labels.extend(labels.numpy())

            # Free memory
            d_input.free()
            d_output.free()

        accuracy = 100 * correct / total
        avg_inference_time = total_time / len(self.test_loader)
        avg_memory_usage = abs(total_memory / len(self.test_loader))  # Ensure non-negative value

        return {
            'model_type': 'TensorRT',
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_memory_usage': avg_memory_usage,
            'predictions': predictions,
            'actual_labels': actual_labels
        }

    def run_complete_profile(self, pytorch_path: str, onnx_path: str, tensorrt_path: str) -> pd.DataFrame:
        """Run complete profiling for all three model types"""
        results = []
        
        # Profile PyTorch model
        print("Profiling PyTorch model...")
        pt_results = self.profile_pytorch_model(pytorch_path)
        results.append(pt_results)
        
        # Profile ONNX model
        print("Profiling ONNX model...")
        onnx_results = self.profile_onnx_model(onnx_path)
        results.append(onnx_results)
        
        # Profile TensorRT model
        print("Profiling TensorRT model...")
        trt_results = self.profile_tensorrt_model(tensorrt_path)
        results.append(trt_results)
        
        # Create DataFrame with results
        df = pd.DataFrame([
            {
                'Model Type': r['model_type'],
                'Accuracy (%)': r['accuracy'],
                'Avg Inference Time (s)': r['avg_inference_time'],
                'Avg Memory Usage (MB)': r['avg_memory_usage']
            }
            for r in results
        ])
        
        return df

