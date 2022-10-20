//
// Created by atrostan on 13/10/22.
//

#ifndef GAPBS_SB_UTILITIES_H
#define GAPBS_SB_UTILITIES_H

#include "benchmark.h"

struct KV {
	uint64_t pr;
	uint64_t id;

	bool operator()(KV const &lhs, KV const &rhs) const {
		if (lhs.pr != rhs.pr)
			return lhs.pr > rhs.pr;
		return lhs.id < rhs.id;
	}
};


class KV_comparator {
	bool reverse;
public:
	KV_comparator(const bool &revparam = false) { reverse = revparam; }

	bool operator()(KV const &lhs, KV const &rhs) const {
		if (lhs.pr != rhs.pr)
			return lhs.pr > rhs.pr;
		return lhs.id < rhs.id;
	}
};

//using namespace folly;
//typedef ConcurrentSkipList<KV, KV_comparator> SkipListType;
//typedef SkipListType::Accessor SkipListAccessor;
//typedef SkipListType::Skipper SkipListSkipper;


std::vector<std::pair<uint64_t, uint64_t>> get_subranges(int n, uint64_t m);

//void par_populate_skiplist(Graph &g, uint64_t n, std::vector<uint64_t> &deg, SkipListAccessor &skiplist);
//
//void par_populate_skiplist(Graph &g, uint64_t n, pvector<uint64_t> &deg, SkipListAccessor &skiplist);
//
//void print_skiplist(SkipListAccessor &sl);

//void par_populate_deg(Graph &g, uint64_t n, pvector<std::pair<uint64_t, uint64_t>> &deg);
void par_populate_deg(Graph &g, uint64_t n, pvector<uint64_t> &deg);



#endif //GAPBS_SB_UTILITIES_H
