module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load cmake/3.15.4-mnqjvz6
#module load netcdf-cxx4/4.3.1-4es6a7y
module load netcdf-cxx4/4.3.1-7lwefeg

export NetCDF_PREFIX=$NETCDF_CXX4_ROOT
cuda=/usr/local/cuda-10.1
#export CMAKE_PREFIX_PATH=$ncxx:$CMAKE_PREFIX_PATH
export THRUST_INCLUDE_DIR=$cuda/include
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH
export OMPI_CXX=`which g++`
export OMPI_CC=`which gcc`
