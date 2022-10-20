//
// Created by atrostan on 13/10/22.
//


#include "par_utils.h"


void mapAdd(std::map<size_t, double> &inout, std::map<size_t, double> &in) {
	for (auto initer = in.begin(), outiter = inout.begin(); initer != in.end(); ++initer, ++outiter) {
		outiter->second += initer->second;
	}
}

/**
 * After computing the Component ID for each vertex in the graph, compute:
 * 1. the unique component IDs in the graph
 * 2. for each component ID, how many vertices are in that component
 * @param comp
 */
void calc_cc_sizes(pvector<uint64_t> &comp, uint64_t n) {
	hash_t map;
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < n; ++i) {
		uint64_t c = comp[i];
		map.try_emplace_l(c,
		                  [](hash_t::value_type &v) { v.second += 1; },
		                  1);
	}
//	fmt::print("map: {}\n", map);
}

void par_decrement(hash_t &decrements, pvector<uint64_t> &deg) {
	// convert to vector for parallel iteration
	std::vector<std::pair<uint64_t, uint64_t>> v{decrements.begin(), decrements.end()};
	uint64_t n = v.size();
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < n; i++) {
		uint64_t vid = v[i].first;
		uint64_t decrement = v[i].second;
		deg[vid] -= decrement;
	}

}

void compute_decrements(pvector<uint64_t> &vids, uint64_t n, Graph &g, hash_t &decrements) {

#pragma omp parallel for schedule(dynamic, 16384)
	for (uint64_t i = 0; i < n; i++) {
		uint64_t u = vids[i];
		uint64_t n_incident_edges = 0;

		for (uint64_t v: g.out_neigh(u)) {
			decrements.try_emplace_l(v,
			                         [](hash_t::value_type &val) { val.second += 1; },
			                         1);
			++n_incident_edges;
		}

		for (uint64_t v: g.in_neigh(u)) {
			decrements.try_emplace_l(v,
			                         [](hash_t::value_type &val) { val.second += 1; },
			                         1);
			++n_incident_edges;
		}

		decrements.try_emplace(u, n_incident_edges);
	}
//	fmt::print("decrements: {}\n", decrements);
}

void par_sort_by_degrees(pvector<uint64_t> &vids, pvector<uint64_t> &degrees) {
	ips4o::parallel::sort(
		vids.begin(),
		vids.end(),
		[&](const uint64_t &a, const uint64_t &b) { return degrees[a] > degrees[b]; });
}

/**
 * given a component assignment to all the yet unassigned permutation id vertices,
 * group all vertices into a container of spokes (connected components that are NOT the
 * giant connected component) sorted by descending size, and sorted by vertex id within each
 * spoke.
 * ignore all the vertices in the giant component
 * @param comp
 * @param bmap
 * @param gcc_id
 */
void par_compute_spokes(pvector<uint64_t> &comp, Bitmap &bmap, uint64_t gcc_id, map_t &spokes,
                        pvector<uint64_t> &perm, uint64_t &spoke_end) {

// populate a map from component id to sets of vertex ids in that component
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < comp.size(); ++i) {
		uint64_t cid = comp[i];
		if (bmap.get_bit(i)) { continue; } // vertex has already been assigned a new id
		else if (cid == gcc_id) { continue; } // vertex is part of GCC, it will be assigned recursively
		else {
			if (spokes.contains(cid)) { spokes[cid].insert(i); }
			else {
				spokes.try_emplace_l(
					cid,
					[&i](map_t::value_type &val) { val.second.insert(i); }, // called only when cid already present in map
					set_t{i}); // otherwise, construct new vertex id set corresponding to cid
			}
		}
	}

	// copy into a vector of vectors and sort
	std::vector<std::vector<uint64_t>> spokes_vec(spokes.size());
	uint64_t spokes_idx = 0;
	uint64_t spoke_start_idx = spoke_end;

	for (auto &kv: spokes) {
		uint64_t n_vs_in_spoke = kv.second.size();
		spokes_vec[spokes_idx].reserve(spokes_vec[spokes_idx].size() + n_vs_in_spoke);
		spokes_vec[spokes_idx].insert(
			spokes_vec[spokes_idx].end(),
			std::make_move_iterator(kv.second.begin()),
			std::make_move_iterator(kv.second.end()));
		// sort by ascending id within each spoke
		std::sort(dpl::execution::par_unseq, spokes_vec[spokes_idx].begin(), spokes_vec[spokes_idx].end());
		++spokes_idx;
		spoke_start_idx -= n_vs_in_spoke;
	}
	std::sort(dpl::execution::par_unseq,
	          spokes_vec.begin(),
	          spokes_vec.end(),
	          [](std::vector<uint64_t> s1, std::vector<uint64_t> s2) -> bool {
		          if (s1.size() == s2.size()) { return s1[0] < s2[0]; }
		          else { return s1.size() > s2.size(); }
	          });

	// compute a partial sum so that each spoke corresponds to an index into the final perm array
	std::vector<uint64_t> perm_start_idcs(spokes_vec.size());
	std::transform(
		dpl::execution::par_unseq,
		spokes_vec.begin(),
		spokes_vec.end(),
		perm_start_idcs.begin(),
		[](std::vector<uint64_t> &l) -> uint64_t { return l.size(); }
	);
	std::exclusive_scan(
		dpl::execution::par_unseq,
		perm_start_idcs.begin(),
		perm_start_idcs.end(),
		perm_start_idcs.begin(),
		spoke_start_idx,
		[](uint64_t l, uint64_t r) -> uint64_t { return l + r; }
	);
	// since spokes are sorted by descending size, omp schedule(static, 1) ensures that division
	// of work among threads is roughly equal
	// todo ^ is this a valid assumption?
#pragma omp parallel for schedule(static, 1)
	for (uint64_t i = 0; i < spokes_vec.size(); ++i) {
		std::vector<uint64_t> &spoke = spokes_vec[i];
		uint64_t k = 0;
		for (uint64_t j = perm_start_idcs[i]; j < perm_start_idcs[i] + spoke.size(); ++j) {
			perm[j] = spoke[k];
			bmap.set_bit_atomic(spoke[k]);
			++k;
		}
	}

	spoke_end = spoke_start_idx;
}

void print_bitmap(Bitmap &bmap, uint64_t size) {
	fmt::print("[");
	for (uint64_t i = 0; i < size - 1; ++i) {
		if (bmap.get_bit(i)) { fmt::print("| "); }
		else { fmt::print("- "); }
	}
	if (bmap.get_bit(size - 1)) { fmt::print("|]\n"); }
	else { fmt::print("-]\n"); }
}

void par_compute_ccs(pvector<uint64_t> &comp, Bitmap &bmap) {
	// a parallel hash map between
	absl::btree_set<uint64_t> s1;
	absl::btree_set<uint64_t> s2;
	s1.merge(s2);

	std::map<int, absl::btree_map<uint64_t, absl::btree_set<uint64_t>>> thread_local_ccs;

	// init the thread local maps
	for (int i = 0; i < omp_get_max_threads(); ++i) { thread_local_ccs[i] = {}; }

	// each thread is assigned a static region of the component assignment vector
	// each thread builds a map between component id -> number of vertices in that component
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < comp.size(); ++i) {
		// skip already assigned vertices
		if (bmap.get_bit(i)) { continue; }
		int t = omp_get_thread_num();
		uint64_t c = comp[i];
		auto &local_cc = thread_local_ccs[t];
		if (!local_cc.contains(c)) { local_cc[c] = {i}; }
		else { local_cc[c].insert(i); }
	}

	fmt::print("thread_local_ccs: {}\n", thread_local_ccs);
}

/**
 * Given a vector of component id assignment, compute in parallel the number of vertices in each
 * component
 * Each open mp thread will compute a thread local absl::flat_hash_map of <component id -> num vertices>
 * Then use absl map.merge to merge all maps
 * i.e. map reduce
 * @param comp
 */
uint64_t par_compute_cc_sizes(pvector<uint64_t> &comp, hash_t &component_counts) {
	std::map<int, absl::flat_hash_map<uint64_t, uint64_t>> thread_local_cc_counts;
	component_counts.clear();

	// init the thread local maps
	for (int i = 0; i < omp_get_max_threads(); ++i) { thread_local_cc_counts[i] = {}; }

	// each thread is assigned a static region of the component assignment vector
	// each thread builds a map between component id -> number of vertices in that component
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < comp.size(); ++i) {
		int t = omp_get_thread_num();
		uint64_t c = comp[i];
		auto &local_cc_count = thread_local_cc_counts[t];
		if (!local_cc_count.contains(c)) { local_cc_count[c] = 1; }
		else { local_cc_count[c] += 1; }
	}

// merge all the thread local component-vertex counts into a global vector of <component id, vertex count in component> pair (to be reduced)
	std::vector<std::pair<uint64_t, uint64_t>> all_counts;
	for (int t = 0; t < omp_get_max_threads(); ++t) {
		all_counts.reserve(all_counts.size() + thread_local_cc_counts[t].size());
		all_counts.insert(
			all_counts.end(),
			std::make_move_iterator(thread_local_cc_counts[t].begin()),
			std::make_move_iterator(thread_local_cc_counts[t].end()));
	}

	// reduce the flattened, thread local component counts by iterating over them in parallel
	// and placing (merging/summing) them in a (thread-safe) parallel hash map
	// at the same time, add component ids to a parallel flat hash set to get the unique component
	// ids for this sb iteration
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < all_counts.size(); ++i) {
		uint64_t cid = all_counts[i].first;
		uint64_t n_vs_in_component = all_counts[i].second;
		component_counts.try_emplace_l(cid,
		                               [&n_vs_in_component](hash_t::value_type &val) { val.second += n_vs_in_component; },
		                               n_vs_in_component);
	}
	auto max_kv = find_max_value_pair(component_counts);
	return max_kv.first; // giant connected component ID
}


std::pair<uint64_t, uint64_t> find_max_value_pair(hash_t &x) {
	using value_t = typename hash_t::value_type;
	const auto compare = [](value_t const &p1, value_t const &p2) {
		return p1.second < p2.second;
	};
	return *std::max_element(dpl::execution::par_unseq, x.begin(), x.end(), compare);;
}


/**
 * sum the sizes of all spokes (ignoring the size of the giant connected component)
 * @param ht
 * @param init
 * @param ignore_key
 * @return
 */
uint64_t accumulate_values(hash_t &ht, uint64_t init, uint64_t ignore_key) {
	return std::transform_reduce(
		dpl::execution::par_unseq,
		ht.begin(), ht.end(), init,
		[](uint64_t l, uint64_t r) { return l + r; },
		[&ignore_key](std::pair<uint64_t, uint64_t> c) -> uint64_t {
			if (c.first != ignore_key) { return c.second; }
			else { return 0; }
		});
}

