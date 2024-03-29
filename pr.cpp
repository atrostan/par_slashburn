//
// Created by atrostan on 14/11/22.
//

// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details


#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "fmt/core.h"
#include "fmt/ranges.h"

/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer
Will return pagerank scores for all vertices once total change < epsilon
This PR implementation uses the traditional iterative approach. It perform
updates in the pull direction to remove the need for atomics, and it allows
new values to be immediately visible (like Gauss-Seidel method). The prior PR
implemention is still available in src/pr_spmv.cc.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;


pvector<ScoreT> PageRankPullGS(const Graph &g, int max_iters,
                               float epsilon = 0) {
	const ScoreT init_score = 1.0f / g.num_nodes();
	const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
	pvector<ScoreT> scores(g.num_nodes(), init_score);
	pvector<ScoreT> outgoing_contrib(g.num_nodes());
#pragma omp parallel for
	for (NodeID n = 0; n < g.num_nodes(); n++)
		outgoing_contrib[n] = init_score / g.out_degree(n);
	for (int iter = 0; iter < max_iters; iter++) {
		float error = 0;
#pragma omp parallel for reduction(+ : error) schedule(dynamic, 16384)
		for (NodeID u = 0; u < g.num_nodes(); u++) {
			ScoreT incoming_total = 0;
			for (NodeID v: g.in_neigh(u))
				incoming_total += outgoing_contrib[v];
			ScoreT old_score = scores[u];
			scores[u] = base_score + kDamp * incoming_total;
			error += fabs(scores[u] - old_score);
			outgoing_contrib[u] = scores[u] / g.out_degree(u);
		}
		printf(" %2d    %lf\n", iter, error);
//		if (error < epsilon)
//			break;
	}
	return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
	vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
	for (NodeID n = 0; n < g.num_nodes(); n++) {
		score_pairs[n] = make_pair(n, scores[n]);
	}
	int k = 5;
	vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
	k = min(k, static_cast<int>(top_k.size()));
	for (auto kvp: top_k)
		cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                float target_error) {
	const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
	pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
	float error = 0;
	for (NodeID u: g.vertices()) {
		ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
		for (NodeID v: g.out_neigh(u))
			incomming_sums[v] += outgoing_contrib;
	}
	for (NodeID n: g.vertices()) {
		error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
		incomming_sums[n] = 0;
	}
	PrintTime("Total Error", error);
	return error < target_error;
}


template<typename T>
void save_pr_scores_as_binary(std::string path, std::vector<T> &scores) {
	std::ofstream ofs(path, std::ios::binary);
	boost::archive::binary_oarchive oa(ofs);
	oa << scores;
}

void read_pr_scores_as_binary(std::string path, std::vector<float> &scores) {
	std::ifstream ifs(path, std::ios::binary);
	boost::archive::binary_iarchive ia(ifs);
	ia >> scores;
}


/**
 * Write a vector<T> to out_path as binary
 * First, writes the size of the vector - .size().
 * Second, writes the vector's data - .data().
 * @tparam T
 * @param out_path
 * @param v
 */
template<typename T>
void write_vector_as_bin(std::string out_path, std::vector<T> &v) {
    std::ofstream out(out_path, std::ios::binary | std::ios::out | std::ios::trunc);
    fmt::print("Writing vector to {}\n", out_path);
    uint64_t size = v.size();
    out.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char *>(v.data()), size * sizeof(T));
    out.close();
}

/**
 * Reads the size of a vector<T> (represented using uint64_t)
 * Fills a dynamic T array[size] with the values read from a binary file
 * Construts a vector<T> from the array, and returns that vector
 * @tparam T
 * @param in_path
 * @return
 */
template<typename T>
std::vector<T> read_vector_as_bin(std::string in_path) {
    std::ifstream in(in_path, std::ios::binary | std::ios::in);
    fmt::print("Reading vector from {}\n", in_path);

    uint64_t size = 0;
    in.read(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    T *arr = new T[size]();
    in.read(reinterpret_cast<char *>(arr), size * sizeof(T));

    std::vector<T> v(arr, arr + size);
    in.close();

    // deallocate before returning
    delete[]arr;
    return v;
}

int main(int argc, char *argv[]) {
	CLPageRank cli(argc, argv, "pagerank", 1e-4, 100);
	if (!cli.ParseArgs())
		return -1;
	Builder b(cli);
	Graph g = b.MakeGraph();

	pvector<ScoreT> scores = PageRankPullGS(g, cli.max_iters(), cli.tolerance());
	auto result = PRVerifier(g, scores, cli.tolerance());
	assert(result);

	// todo make consistent datatype of vector (double, float) for future io of binary files
	// std::vector<float> pr_values(scores.data(), scores.end());
	std::vector<double> pr_values(scores.data(), scores.end());
	// fmt::print("pr_values: {}\n", pr_values);
	// save_pr_scores_as_binary<double>(cli.pr_output_path(), pr_values);
	write_vector_as_bin<double>(cli.pr_output_path(), pr_values);
	// std::vector<float> read_pr_values;
	// read_pr_scores_as_binary(cli.pr_output_path(), read_pr_values);
	// fmt::print("read_pr_values: {}\n", read_pr_values);
	
	return 0;
}