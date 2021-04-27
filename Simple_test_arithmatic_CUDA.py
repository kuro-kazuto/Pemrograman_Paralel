import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


#Host variables
a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
b = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
k = np.float32(2.0)

#Device Variables
a_d = cuda.mem_alloc(a.nbytes)
b_d = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(a_d, a)
cuda.memcpy_htod(b_d, b)
s_d = cuda.mem_alloc(a.nbytes)
m_d = cuda.mem_alloc(a.nbytes)

#Device Source
mod = SourceModule("""
    __global__ void S(float *s, float *a, float *b)
    {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * blockDim.y + ty;
        int col = bx * blockDim.x + tx;
        int dim = gridDim.x * blockDim.x;
        const int id = row * dim + col;
        s[id] = a[id] + b[id];
    }

    __global__ void M(float *m, float *a, float k)
    {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * blockDim.y + ty;
        int col = bx * blockDim.x + tx;
        int dim = gridDim.x * blockDim.x;
        const int id = row * dim + col;
        m[id] = k * a[id];
    }
""")

#Vector addition
func = mod.get_function("S")
func(s_d, a_d, b_d, block=(1,3,1))
s = np.empty_like(a)
cuda.memcpy_dtoh(s, s_d)

#Vector multiplication by constant
func = mod.get_function("M")
func(m_d, a_d, k, block=(1,3,1))
m = np.empty_like(a)
cuda.memcpy_dtoh(m, m_d)

print ("Vector Addition")
print ("Expected: " + str(a+b))
print ("Result: " + str(s) + "\n")
print ("Vector Multiplication")
print ("Expected: " + str(k*a))
print ("Result: " + str(m))