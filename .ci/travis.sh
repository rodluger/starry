# Miniconda (cached)
export PATH="$HOME/miniconda-cache/bin:$PATH"
if [ ! -f $HOME/miniconda-cache/bin/conda ]; then
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
      bash miniconda.sh -b -p $HOME/miniconda-cache -u;
      conda config --add channels conda-forge;
      conda config --set always_yes yes;
      conda update --all;
      conda create --yes -n test python=$PYTHON_VERSION
      conda activate test
      conda install -c conda-forge \
            numpy=$NUMPY_VERSION \
            scipy \
            matplotlib \
            setuptools \
            pybind11 \
            pytest \
            pytest-cov \
            pip \
            healpy\
            nbsphinx \
            theano;
      pip install Pillow
      pip install ipython
      pip install jupyter
      pip install sphinx
      pip install git+git://github.com/rodluger/starry_beta.git
fi

# Display some info
conda info -a