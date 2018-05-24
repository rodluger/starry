# Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n test python=$PYTHON_VERSION
conda activate test
conda install -c conda-forge numpy=$NUMPY_VERSION scipy matplotlib setuptools pybind11 pytest pytest-cov pip healpy nbsphinx
pip install Pillow
pip install batman-package
pip install tqdm
pip install starry_maps
pip install ipython
pip install jupyter
pip install emcee

# Build the code
# Build the extension
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    python setup.py develop
else
    CXX=g++-4.8 CC=gcc-4.8 python setup.py develop
fi
