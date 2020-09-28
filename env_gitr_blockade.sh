module purge
module load gcc
module load mpich
module load cmake/3.15.4-kgiql7d
module load netcdf
module load vim

ncxx=/lore/gopan/install/build-netcdfcxx431/install
export NetCDF_CXX4_PREFIX=$ncxx
export CMAKE_PREFIX_PATH=$ncxx:$CMAKE_PREFIX_PATH
cuda=/usr/local/cuda-10.1
export THRUST_INCLUDE_DIR=$cuda/include
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH

#export  NETCDFLIB=netcdf
#export  NETCDFLIB_CPP=libnetcdf-cxx4.so
#libconfig=/lore/gopan/install/libconfig-1.7.2/install
#libconf=/lore/gopan/install/build-libconfig-1.7.2/install
libconf=/lore/gopan/install/libconfig-1.7.2-dist/install
export LIB_CONFIG_INSTALL=$libconf
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${ncxx}/lib64/pkgconfig
#export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${libconf}/lib64/pkgconfig
#export CMAKE_PREFIX_PATH=${libconf}:$CMAKE_PREFIX_PATH
