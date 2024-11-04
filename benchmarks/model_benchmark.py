# benchmarks/model_benchmark.py

import time
import statistics
import json
from flask import Flask
import os
from pathlib import Path
import tempfile
import shutil
from io import BytesIO
from werkzeug.datastructures import FileStorage

# Add project root to Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from app.models import FaceModel, ArmModel, SpeechModel
from app.api.routes import api_bp

class ModelBenchmark:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(api_bp, url_prefix='/api')
        self.client = self.app.test_client()
        
        # Initialize models
        self.face_model = FaceModel()
        self.arm_model = ArmModel()
        self.speech_model = SpeechModel()
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent
        self.test_examples_dir = self.project_root / 'tests'/ 'examples'
        self.results_dir = self.project_root / 'benchmarks' / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Test file paths
        self.positive_csv = self.test_examples_dir / 'positive_sample_arm.csv'
        self.positive_audio = self.test_examples_dir / 'positive_sample_audio.wav'
        self.positive_image = self.test_examples_dir / 'positive_sample_face.jpg'
        
        # Load file contents into memory
        self._load_test_files()
        
        # Verify test files exist
        self._verify_test_files()
    
    def _verify_test_files(self):
        """Verify all required test files exist"""
        required_files = [
            (self.positive_csv, "positive_sample_arm.csv"),
            (self.positive_audio, "positive_sample_audio.wav"),
            (self.positive_image, "positive_sample_face.jpg")
        ]
        
        missing_files = [str(file[1]) for file in required_files if not file[0].exists()]
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing test files in {self.test_examples_dir}:\n" +
                "\n".join(missing_files)
            )
    
    def _load_test_files(self):
        """Load test files into memory"""
        # Load image data
        with open(self.positive_image, 'rb') as f:
            self.image_data = f.read()
            
        # Load CSV data
        with open(self.positive_csv, 'rb') as f:
            self.csv_data = f.read()
            
        # Load audio data
        with open(self.positive_audio, 'rb') as f:
            self.audio_data = f.read()

    def create_file_storage(self, file_content, filename, content_type):
        """Create a FileStorage object from file content"""
        return FileStorage(
            stream=BytesIO(file_content),
            filename=filename,
            content_type=content_type
        )

    def measure_inference_time(self, func, iterations=100):
        times = []
        warm_up_iterations = 5  # Warm-up iterations to stabilize performance
        
        # Warm-up phase
        print("Warming up...")
        for _ in range(warm_up_iterations):
            func()
        
        # Actual measurement
        print(f"Running {iterations} iterations...")
        for i in range(iterations):
            if i % 10 == 0:  # Progress indication every 10 iterations
                print(f"Progress: {i}/{iterations}")
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std_dev': statistics.stdev(times),
            'min': min(times),
            'max': max(times),
            'samples': iterations
        }
    
    def benchmark_face_model(self):
        def run_inference():
            file_storage = self.create_file_storage(
                self.image_data,
                self.positive_image.name,
                'image/jpeg'
            )
            self.face_model.predict(file_storage)
                
        return self.measure_inference_time(run_inference)
    
    def benchmark_arm_model(self):
        def run_inference():
            file_storage = self.create_file_storage(
                self.csv_data,
                self.positive_csv.name,
                'text/csv'
            )
            self.arm_model.predict(file_storage)
                
        return self.measure_inference_time(run_inference)
    
    def benchmark_speech_model(self):
        def run_inference():
            file_storage = self.create_file_storage(
                self.audio_data,
                self.positive_audio.name,
                'audio/wav'
            )
            self.speech_model.predict(file_storage)
                
        return self.measure_inference_time(run_inference)
    
    def run_all_benchmarks(self):
        print("\nRunning benchmarks (this may take a few minutes)...")
        results = {}
        
        # Run benchmarks one by one with progress indication
        print("\nBenchmarking Face Model...")
        results['face_model'] = self.benchmark_face_model()
        
        print("\nBenchmarking Arm Model...")
        results['arm_model'] = self.benchmark_arm_model()
        
        print("\nBenchmarking Speech Model...")
        results['speech_model'] = self.benchmark_speech_model()
        
        # Print results in a formatted way
        print("\nModel Inference Time Benchmarks")
        print("=" * 50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}")
            print("-" * 30)
            print(f"Mean time:     {metrics['mean']*1000:.2f} ms")
            print(f"Median time:   {metrics['median']*1000:.2f} ms")
            print(f"Std dev:       {metrics['std_dev']*1000:.2f} ms")
            print(f"Min time:      {metrics['min']*1000:.2f} ms")
            print(f"Max time:      {metrics['max']*1000:.2f} ms")
            print(f"Sample size:   {metrics['samples']}")
        
        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f'benchmark_results_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\nResults saved to: {result_file}")
        return results

if __name__ == '__main__':
    try:
        print("Initializing benchmark...")
        benchmark = ModelBenchmark()
        print("Starting benchmarks...")
        benchmark.run_all_benchmarks()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Error details: {type(e).__name__}, {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)