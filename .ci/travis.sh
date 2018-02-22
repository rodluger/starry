# Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update --all
conda info -a
#conda create --yes -n test python=$PYTHON_VERSION
#source activate test
conda install -c conda-forge numpy=$NUMPY_VERSION scipy matplotlib sympy mpmath setuptools pytest pytest-cov pip healpy PIL

# Build the code
python setup.py install
