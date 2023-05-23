# HEOA

On a machine with a GPU, please install JAX in a virtual environment using the following command, which will ALSO install a compatible CUDA toolkit:
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Make sure your machine's drivers meet the minimum required driver version:
https://docs.nvidia.com/deploy/cuda-compatibility/index.html
