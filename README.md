# brax_soccer

# install
- nvidia-driver >= 525
~~~
sudo apt-get purge nvidia*
sudo apt-get install nvidia-driver-525
sudo reboot
~~~
- install jax
~~~
conda install jax -c conda-forge
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
~~~
-install cuda-nvcc (must be same with cudatookit version)
~~~
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
~~~
-install brax from source(0.1.1)
~~~
git clone https://github.com/google/brax.git
cd brax
pip install -e .
~~~
