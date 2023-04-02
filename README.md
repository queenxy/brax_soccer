# brax_soccer

# install
- nvidia-driver >= 525
- install jax
~~~
conda install jax -c conda-forge
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
~~~
-install cuda-nvcc (must be same with cudatookit version)
~~~
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
~~~
-install brax from source
~~~
git clone https://github.com/google/brax.git
cd brax
pip install -e .
~~~
