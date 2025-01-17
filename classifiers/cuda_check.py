import cudf
from numba import cuda

# Create a simple DataFrame
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)

# Check GPU functionality
@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] + y[i]

x = cuda.to_device(df['a'].to_array())
y = cuda.to_device(df['b'].to_array())
out = cuda.device_array_like(x)
add_kernel.forall(x.size)(x, y, out)

print(out.copy_to_host())

