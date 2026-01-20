set -ex

cd test/compare_with_liblbfgs/liblbfgs
./autogen.sh
./configure
make
sudo make install  # To install libLBFGS library and header.
cd ../../../

g++ -o test/compare_with_liblbfgs/make_data test/compare_with_liblbfgs/make_data.cpp -L/usr/local/lib -llbfgs
mkdir -p test/compare_with_liblbfgs/data
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./test/compare_with_liblbfgs/make_data test/compare_with_liblbfgs/data
