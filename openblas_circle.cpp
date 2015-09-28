#include "lapacke.h"

#include <vector>
#include <random>
#include <chrono>
#include <iostream>

void invert_in_place(double* A, int m, int n)
{
	std::cout << "inverting " << m << "x"<< n << " matrix using dgetrf/dgetri" << std::endl;

	int lda = m;
	int lwork = 3 * std::max(1, n + n);
	int info = 0;

	std::vector<double> work(lwork);
	std::vector<int> ipiv(m);

	LAPACK_dgetrf(&m, &n, A, &lda, ipiv.data(), &info);
	LAPACK_dgetri(&n, A, &lda, ipiv.data(), work.data(), &lwork, &info);
}

void fill_rand(double* A, int m, int n)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(1, 10);
	int length = m*n;
	for (size_t i = 0; i < length; ++i)
		A[i] = dis(gen);
}

int main(int argc, char* argv[])
{
	using namespace std::chrono;

	int mat_size = 256;

	std::vector<double> test_matrix(mat_size*mat_size);

	fill_rand(test_matrix.data(), mat_size, mat_size);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	invert_in_place(test_matrix.data(), mat_size, mat_size);

	auto invert_time = high_resolution_clock::now() - start;
	auto us = duration_cast<microseconds>(invert_time).count();
	std::cout << "elapsed time " << us << "us" << std::endl;
}

