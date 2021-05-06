# Source: https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line

# If needed, uninstall CMAKE
sudo apt remove --purge --auto-remove cmake

# Specify build, download, and extract
version=3.17
build=1
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/

# Install CMAKE
./bootstrap
make -j$(nproc)
sudo make install

# Once finished, source environment and test cmake
source ~/.bashrc
cmake --version
