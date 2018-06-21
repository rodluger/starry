# Miniconda (cached)
export PATH="$HOME/miniconda/bin:$PATH"
if ! command -v conda > /dev/null; then
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
      bash miniconda.sh -b -p $HOME/miniconda -u;
      conda config --add channels conda-forge;
      conda config --set always_yes yes;
      conda update --all;
      conda create --yes -n test python=$PYTHON_VERSION
      conda activate test
      conda install tectonic;
      conda install -c conda-forge numpy=$NUMPY_VERSION scipy matplotlib setuptools pybind11 pytest pytest-cov pip healpy nbsphinx;
      pip install Pillow
      pip install batman-package
      pip install tqdm
      pip install ipython
      pip install jupyter
      pip install emcee
      pip install corner
      pip install git+git://github.com/tomlouden/SPIDERMAN.git@69911b042bc46615ec9b39048a69e0d77c8542ad
      pip install sphinx
fi

# DEBUG
sudo apt-get install fonts-lmodern
ls usr/share/fonts/truetype
rm -rf ~/.cache/matplotlib/fontList.cache
python -c "import matplotlib.font_manager; print('\n'.join([font for font in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')]))"
python -c "import matplotlib.font_manager; flist = matplotlib.font_manager.get_fontconfig_fonts(); names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]; print(names)"

# DEBUG
exit

# Install starry_maps
pip install starry_maps

# Display some info
conda info -a

# Build the code
CC=gcc-5 && CXX=g++-5 python setup.py develop
