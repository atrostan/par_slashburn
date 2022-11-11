//
// Created by atrostan on 16/10/22.
//

#ifndef GAPBS_SB_SLASHBURN_H
#define GAPBS_SB_SLASHBURN_H

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#include "benchmark.h"
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/synchronization/mutex.h"
#include "parallel-hashmap/parallel_hashmap/phmap.h"
#include "parallel-hashmap/parallel_hashmap/btree.h"

#include "pvector.h"
#include "benchmark.h"
#include "omp.h"
#include "ips4o/ips4o.hpp"
#include "bitmap.h"

#define MAPNAME phmap::parallel_flat_hash_map
#define NMSP phmap
#define MTX absl::Mutex
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

using absl_set_t = absl::flat_hash_set<uint64_t>;

template<class K, class V>
using HashT = MAPNAME<K, V EXTRAARGS>;
using hash_t = HashT<uint64_t, uint64_t>;
using map_t = HashT<uint64_t, set_t>;

class SlashBurn {
public:
	Graph &g;
	int n_neighbour_rounds;
	int k;
	float prec;

	uint64_t num_nodes;
	pvector<uint64_t> degrees;
	pvector<uint64_t> vids;
	pvector<uint64_t> top_k;
	pvector<uint64_t> perm;
	pvector<uint64_t> comp;

	uint64_t time;
	uint64_t hub_start;
	uint64_t spokes_end;
	uint64_t gcc_id;
	uint64_t gcc_size;
	uint64_t n_iters;

	map_t spokes;
	hash_t component_sizes;
	hash_t decrements;
	Bitmap &bmap;

	pvector<uint64_t> active_vertex_set;
    uint64_t active_vertex_set_size;

	SlashBurn(Graph &g, int n_neighs, float prec, Bitmap &bitmap);

	void remove_from_active_vertex_set(pvector<uint64_t> &vs_to_remove);

	void get_active_vertex_set();

	void get_final_components(map_t &comps);

	void convert_vertex_set_to_vec();

	void init();

	void clear();

	void populate_degrees();

	void sort_by_degrees();

	bool CCVerifier();

	void Link(NodeID u, NodeID v);

	void Compress();

	void Afforest();

	NodeID SampleFrequentElement(int64_t num_samples);

	void par_compute_cc_sizes();

	void PrintCompStats();

	void remove_k_hubs();

	void par_compute_spokes();

	void compute_decrements(pvector<uint64_t> &vids);

	void par_decrement();

	std::pair<uint64_t, uint64_t> find_max_value_pair(hash_t &x);

	void print_bitmap();

	void write_permutation(std::string path);

	void place_final_comp_vec(map_t & comps);

    bool vertex_inactive(uint64_t vid);

    void get_set(pvector<uint64_t> &p, std::set<uint64_t> &s) ;
};


#endif //GAPBS_SB_SLASHBURN_H
