#!/usr/bin/env python3
"""
CUDA Availability and System Information Checker
"""

import sys
import platform
import subprocess

def check_python_info():
    """Check Python version and installation"""
    print("=" * 60)
    print("PYTHON INFORMATION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print("\n" + "=" * 60)
    print("PYTORCH INFORMATION")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch Install Path: {torch.__file__}")
        
        # Check CUDA support in PyTorch
        print(f"\nCUDA Support Built-in: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {device.name}")
                print(f"  - Memory: {device.total_memory / 1024**3:.2f} GB")
                print(f"  - Compute Capability: {device.major}.{device.minor}")
                print(f"  - Multiprocessors: {device.multi_processor_count}")
        else:
            print("CUDA is not available in PyTorch!")
            
        # Check current device
        print(f"\nDefault Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
        # Test tensor creation
        try:
            if torch.cuda.is_available():
                x = torch.tensor([1.0, 2.0]).cuda()
                print(f"CUDA Tensor Test: SUCCESS - {x.device}")
            else:
                x = torch.tensor([1.0, 2.0])
                print(f"CPU Tensor Test: SUCCESS - {x.device}")
        except Exception as e:
            print(f"Tensor Test: FAILED - {e}")
            
    except ImportError:
        print("PyTorch is not installed!")
        print("Install with: pip install torch torchvision")

def check_nvidia_driver():
    """Check NVIDIA driver and system CUDA"""
    print("\n" + "=" * 60)
    print("NVIDIA SYSTEM INFORMATION")
    print("=" * 60)
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("NVIDIA Driver: INSTALLED")
            print("\nnvidia-smi output:")
            print("-" * 40)
            print(result.stdout)
        else:
            print("NVIDIA Driver: NOT FOUND or ERROR")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi: NOT FOUND")
        print("NVIDIA drivers may not be installed")
    except subprocess.TimeoutExpired:
        print("nvidia-smi: TIMEOUT")
    except Exception as e:
        print(f"Error checking nvidia-smi: {e}")
    
    # Check NVCC (CUDA compiler)
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\nCUDA Toolkit (nvcc): INSTALLED")
            print(result.stdout)
        else:
            print("\nCUDA Toolkit (nvcc): NOT FOUND")
    except FileNotFoundError:
        print("\nCUDA Toolkit (nvcc): NOT FOUND")
    except Exception as e:
        print(f"\nError checking nvcc: {e}")

def check_environment_variables():
    """Check CUDA-related environment variables"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    import os
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'PATH', 'LD_LIBRARY_PATH']
    
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH':
            # Check if CUDA is in PATH
            paths = value.split(os.pathsep) if value != 'Not set' else []
            cuda_in_path = any('cuda' in path.lower() for path in paths)
            print(f"{var}: {'CUDA found in PATH' if cuda_in_path else 'No CUDA in PATH'}")
        else:
            print(f"{var}: {value}")

def run_simple_test():
    """Run a simple CUDA computation test"""
    print("\n" + "=" * 60)
    print("SIMPLE CUDA TEST")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available - running CPU test")
            device = torch.device('cpu')
        else:
            print("CUDA available - running GPU test")
            device = torch.device('cuda')
        
        # Create test tensors
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Time matrix multiplication
        import time
        start_time = time.time()
        c = torch.mm(a, b)
        end_time = time.time()
        
        print(f"Matrix multiplication (1000x1000) on {device}:")
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"Result shape: {c.shape}")
        print(f"Result device: {c.device}")
        print("Test: SUCCESS")
        
    except Exception as e:
        print(f"Test: FAILED - {e}")

def check_common_issues():
    """Check for common CUDA issues"""
    print("\n" + "=" * 60)
    print("COMMON ISSUES DIAGNOSIS")
    print("=" * 60)
    
    issues = []
    
    try:
        import torch
        
        # Check if PyTorch was compiled with CUDA
        if not torch.cuda.is_available():
            issues.append("PyTorch was not compiled with CUDA support")
        
        # Check if NVIDIA driver is installed
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError):
            issues.append("NVIDIA drivers not installed or not working")
        
        # Check PyTorch CUDA version compatibility
        if torch.cuda.is_available():
            try:
                torch_cuda_version = torch.version.cuda
                print(f"PyTorch compiled with CUDA: {torch_cuda_version}")
            except:
                issues.append("Could not determine PyTorch CUDA version")
        
    except ImportError:
        issues.append("PyTorch not installed")
    
    if issues:
        print("POTENTIAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\nSOLUTIONS:")
        print("1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("2. Install CUDA-enabled PyTorch:")
        print("   Visit: https://pytorch.org/get-started/locally/")
        print("   Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("3. Check if your GPU supports CUDA (compute capability >= 3.5)")
        print("4. Restart your computer after driver installation")
    else:
        print("No common issues detected!")

def main():
    """Main function to run all checks"""
    print("CUDA AVAILABILITY AND SYSTEM CHECKER")
    print("=" * 60)
    
    check_python_info()
    check_pytorch()
    check_nvidia_driver()
    check_environment_variables()
    run_simple_test()
    check_common_issues()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is AVAILABLE and working!")
            print(f"✅ {torch.cuda.device_count()} GPU(s) detected")
            print("✅ You can use GPU acceleration")
        else:
            print("❌ CUDA is NOT AVAILABLE")
            print("❌ Training will use CPU only (much slower)")
            print("💡 Consider using CPU-only mode or fix CUDA installation")
    except ImportError:
        print("❌ PyTorch not installed")
    
    print("\nFor CPU-only training, use:")
    print("python training_script.py --data_dir ./data --device cpu")

if __name__ == "__main__":
    main()