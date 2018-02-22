# Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -n test python=$PYTHON_VERSION
conda activate test
conda install -c conda-forge numpy=$NUMPY_VERSION scipy matplotlib sympy mpmath setuptools pytest pytest-cov pip healpy
pip install Pillow

# Build the code
python setup.py install

# Generate the figures
echo "Generating figures..."
cd tex/figures
for f in *.py; do
    python "$f"
done
