#!/bin/bash
cd /tmp

# install unzip
sudo apt-get install unzip -y

# create directory for cache
sudo mkdir -p starryfonts
cd starryfonts

# download fonts
if [ ! -f FontAwesome.zip ]; then
    wget "https://github.com/FortAwesome/Font-Awesome/archive/v4.7.0.zip" -O FontAwesome.zip
fi

# extract fonts
sudo unzip FontAwesome.zip -o -d FontAwesome

# create target directories for fonts
sudo mkdir -p /usr/share/fonts/truetype/FontAwesome
sudo mkdir -p /usr/share/fonts/opentype/FontAwesome

# copy fonts
sudo cp FontAwesome/Font-Awesome*/fonts/fontawesome-webfont.ttf /usr/share/fonts/truetype/FontAwesome/fontawesome.ttf
sudo cp FontAwesome/Font-Awesome*/fonts/FontAwesome.otf /usr/share/fonts/opentype/FontAwesome/

# build cache
sudo fc-cache -fv

# return
cd $TRAVIS_BUILD_DIR
