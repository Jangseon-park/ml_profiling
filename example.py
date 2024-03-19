import torch
import numpy as np
from torch.profiler import ExecutionTraceObserver, profile

def trace_handler(prof):
    prof.export_chrome_trace("kineto_trace.json")

def gpu_matrix_multiplication(matrix1: np.ndarray, matrix2: np.ndarray) -> torch.Tensor:
    """
    Perform matrix multiplication on the GPU using PyTorch.

    Args:
        matrix1 (np.ndarray): The first input matrix as a NumPy array.
        matrix2 (np.ndarray): The second input matrix as a NumPy array.

    Returns:
        torch.Tensor: The result of the matrix multiplication, as a PyTorch tensor.

    Raises:
        ValueError: If matrices have incompatible shapes for multiplication.
    """
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("Matrices have incompatible shapes for multiplication.")

    # Convert numpy arrays to PyTorch tensors and set dtype to float
    matrix1_torch = torch.tensor(matrix1, dtype=torch.float)
    matrix2_torch = torch.tensor(matrix2, dtype=torch.float)

    # Transfer tensors to GPU if available
    if torch.cuda.is_available():
        matrix1_torch = matrix1_torch.to('cuda')
        matrix2_torch = matrix2_torch.to('cuda')

    # Perform matrix multiplication using GPU
    result_gpu = torch.matmul(matrix1_torch, matrix2_torch)

    return result_gpu

if __name__ == "__main__":
    et = ExecutionTraceObserver()
    et.register_callback("pytorch_et.json")

    # Define larger matrices (1024x1024) using NumPy
    matrix_a = np.random.rand(1024, 1024)
    matrix_b = np.random.rand(1024, 1024)

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=10, active=1),
        on_trace_ready=trace_handler
    ) as prof:
        for epoch in range(20):
            result_on_gpu = gpu_matrix_multiplication(matrix_a, matrix_b)
            if epoch == 11:
                et.stop()
            if epoch == 10:
                et.start()
            prof.step()

    et.unregister_callback()