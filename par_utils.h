//
// Created by atrostan on 13/10/22.
//
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"

#include "parallel-hashmap/parallel_hashmap/phmap.h"
#include "parallel-hashmap/parallel_hashmap/btree.h"
#include "SlashBurn.h"

#include "pvector.h"
#include "benchmark.h"
#include "omp.h"
#include "ips4o/ips4o.hpp"
#include "bitmap.h"

#ifndef GAPBS_SB_PAR_UTILS_H
#define GAPBS_SB_PAR_UTILS_H


#define MAPNAME phmap::parallel_flat_hash_map
#define NMSP phmap
#define MTX std::mutex
#define EXTRAARGS                                                       \
    , NMSP::priv::hash_default_hash<K>, NMSP::priv::hash_default_eq<K>, \
        std::allocator<std::pair<const K, V>>, 4, MTX

#define SETNAME phmap::parallel_flat_hash_set
#define SETEXTRAARGS \
, NMSP::priv::hash_default_hash<K>, NMSP::priv::hash_default_eq<K>, \
        std::allocator<K>, 4, MTX

template<class K>
using SetT = SETNAME<K SETEXTRAARGS>;
using set_t = SetT<uint64_t>;

template<class K, class V>
using HashT = MAPNAME<K, V EXTRAARGS>;
using hash_t = HashT<uint64_t, uint64_t>;
//using map_t = HashT<uint64_t, set_t>;

void calc_cc_sizes(pvector<uint64_t> &comp, uint64_t n);

void compute_decrements(pvector<uint64_t> &vids, uint64_t n, Graph &g, hash_t &decrements);

void par_decrement(hash_t &decrements, pvector<uint64_t> &deg);

void par_sort_by_degrees(pvector<uint64_t> &vids, pvector<uint64_t> &degrees);

uint64_t par_compute_cc_sizes(pvector<uint64_t> &comp, hash_t &component_sizes);

void par_compute_ccs(pvector<uint64_t> &comp, Bitmap &bmap);

std::pair<uint64_t, uint64_t> find_max_value_pair(hash_t &x);

uint64_t accumulate_values(hash_t &ht, uint64_t init, uint64_t ignore_key);

void par_compute_spokes(pvector<uint64_t> &comp, Bitmap &bmap, uint64_t gcc_id, map_t &spokes,
                        pvector<uint64_t> &perm, uint64_t &spoke_end);

void print_bitmap(Bitmap &bmap, uint64_t size);

template<typename T>
void flatten(const std::vector<std::vector<T>> &v, std::vector<T> &result) {
	std::size_t total_size = 0;
	for (const auto &sub: v)
		total_size += sub.size(); // I wish there was a transform_accumulate
	result.reserve(total_size);
	for (const auto &sub: v)
		result.insert(result.end(), sub.begin(), sub.end());
}

#endif //GAPBS_SB_PAR_UTILS_H
