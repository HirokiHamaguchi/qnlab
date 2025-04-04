g++ -o test/compare_with_liblbfgs/make_data test/compare_with_liblbfgs/make_data.cpp -llbfgs
mkdir -p test/compare_with_liblbfgs/data
./test/compare_with_liblbfgs/make_data test/compare_with_liblbfgs/data
