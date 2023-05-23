# HEOA

On a Linux machine with a GPU, please install JAX in a virtual environment using the following command, which will ALSO install a compatible CUDA toolkit:
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Make sure your machine's drivers meet the minimum required driver version:
https://docs.nvidia.com/deploy/cuda-compatibility/index.html.

Next, install OTT-JAX using:
```
pip install ott-jax==0.3.1
```
Lastly, navigate to your `moscot_not` directory (which Merel has to provide) and install the package and its dependencies using:
```
pip install -e .
```
If you get the following error, try upgrading pip first:
``` 
ERROR: File "setup.py" or "setup.cfg" not found.
```
Now, your environment contains all the packages (including `anndata`, `scanpy` and plotting libraries) needed to run the OT notebooks.
