#include "lapacke.h"
#include "cblas.h"
#include "external/ThreadPool.h"

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <list>

using namespace std::chrono;


/**
 * Calls the getrf/getri lapack routines to invert a matrix of doubles. With threading enabled this takes ~2000 times longer than without (12s vs 5ms).
 */
void invert_in_place(double* A, int m, int n, bool verbose = true)
{
	if (verbose)
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

int invert_random_matrix(int mat_size, bool verbose = true)
{
	std::vector<double> test_matrix(mat_size*mat_size);

	fill_rand(test_matrix.data(), mat_size, mat_size);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	invert_in_place(test_matrix.data(), mat_size, mat_size, verbose);

	auto invert_time = high_resolution_clock::now() - start;
	auto us = duration_cast<microseconds>(invert_time).count();

	if (verbose)
		std::cout << "elapsed time " << us << "us" << std::endl;

	return us;
}

/**
 * Uses Jakob Progsch's ThreadPool implementation to invert a matrix in several threads, which triggers a segfault when run on a circleci container.
 */
void threaded_invert(int mat_size)
{
	std::list<std::future<int>> results;
	int thread_count = 7;
	ThreadPool pool(thread_count);

	for (int i = 0; i < thread_count; ++i)
	{
		results.emplace_back(pool.enqueue([](int mat_size) -> int
		{
			return invert_random_matrix(mat_size, false);
		}, mat_size));
	}


	int thread = 0;
	for (auto& result : results) {
		std::cout << "result time from thread " << thread << ": " << result.get() << "us" << std::endl;
		thread++;
	}
}

int main(int argc, char* argv[])
{
	int mat_size = 256;
	invert_random_matrix(mat_size);

	std::cout << "changing openblas thread count from " << openblas_get_num_threads() << " to 1" << std::endl;
	openblas_set_num_threads(1);

	invert_random_matrix(mat_size);

	threaded_invert(mat_size);
}

