# Install gtsam
set -x
CWD=`pwd`
git clone --depth 1 https://github.com/borglab/gtsam gtsam

cd gtsam

pip install -r cython/requirements.txt

mkdir build
cd build

cmake -DGTSAM_ALLOW_DEPRECATED_SINCE_V4=OFF \
      -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
      -DGTSAM_INSTALL_CYTHON_TOOLBOX=ON \
      -DCMAKE_INSTALL_PREFIX=$CWD/gtsam_bin .. && make -j$(nproc) && make install

cd ../..
