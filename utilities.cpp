//
// Created by atrostan on 13/10/22.
//

#include "utilities.h"
#include "benchmark.h"
#include <omp.h>
#include <future>
#include <fmt/core.h>
#include <fmt/ranges.h>


std::vector<std::pair<uint64_t, uint64_t>> get_subranges(int n, uint64_t m) {
	// return n (almost equal) sub-ranges of the range defined by [0, m)
	uint64_t n_els_per_range = m / n;
	std::vector<std::pair<uint64_t, uint64_t>> subs(n);
	for (uint64_t i = 0; i < n; ++i) {
		subs[i] = std::make_pair(i * n_els_per_range, (i + 1) * n_els_per_range);
	}
	subs[n - 1].second = m;
	return subs;
}
//
//
//void print_skiplist(SkipListAccessor &sl) {
//
//	std::vector<std::pair<uint64_t, uint64_t>> to_print(sl.size());
//	uint64_t j = 0;
//	for (auto i = sl.begin(); i != sl.end(); ++i) {
//		to_print[j].first = i->id;
//		to_print[j].second = i->pr;
//		++j;
//	}
//	fmt::print("SkipList (ID, Priority):\n {}\n", to_print);
//}
//
//void par_populate_skiplist(Graph &g, uint64_t n, pvector<uint64_t> &deg, SkipListAccessor &skiplist) {
//#pragma omp parallel for schedule(static)
//	for (uint64_t v = 0; v < g.num_nodes(); ++v) {
//		uint64_t d = g.out_degree(v) + g.in_degree(v);
//		deg[v] = d;
////		skiplist.insert(KV{d, v});
//	}
//}
//
//void par_populate_skiplist(Graph &g, uint64_t n, std::vector<uint64_t> &deg, SkipListAccessor &skiplist) {
//	int n_threads = omp_get_max_threads();
//
//	std::vector<std::pair<uint64_t, uint64_t>> sub_ranges = get_subranges(n_threads, n);
//	auto thread_main = [&](uint64_t start, uint64_t end) {
//		for (uint64_t v = start; v < end; ++v) {
//			uint64_t d = g.out_degree(v) + g.in_degree(v);
//			deg[v] = d;
//			skiplist.insert(KV{d, v});
//		}
//	};
//	std::vector<std::future<void>> futures;
//	for (uint64_t i = 0; i < n_threads; ++i) {
//		uint64_t start = sub_ranges[i].first;
//		uint64_t end = sub_ranges[i].second;
//		futures.push_back(std::async(thread_main, start, end));
//	}
//	for (auto &f: futures) {
//		f.get();
//	}
//}

void par_populate_deg(Graph &g, uint64_t n, pvector<uint64_t> &deg) {
#pragma omp parallel for schedule(static)
	for (uint64_t v = 0; v < g.num_nodes(); ++v) {
		uint64_t d = g.out_degree(v) + g.in_degree(v);
		deg[v] = d; // pairs of {vertex id, current degree in graph}
//		skiplist.insert(KV{d, v});
	}
}