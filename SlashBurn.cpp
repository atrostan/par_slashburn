//
// Created by atrostan on 16/10/22.
//

#include "SlashBurn.h"

typedef std::chrono::duration<long long, std::milli> time_unit;

SlashBurn::SlashBurn(Graph &g, int n_neighs, float p, Bitmap &bitmap) : g(g), bmap(bitmap) {
	prec = p;
	n_neighbour_rounds = n_neighs;
	num_nodes = g.num_nodes();
	gcc_size = num_nodes;

	k = num_nodes * prec;
	if (k == 0) { k = 1; }
	init();
	fmt::print("k: {}\n", k);
	auto start = std::chrono::high_resolution_clock::now();
	n_iters = 0;
	while (true) {
		sort_by_degrees();
        // if, after sorting, the highest degree vertex is a singleton, the graph is completely fragmented, so break
        if (degrees[vids[0]]== 0) break;
		remove_k_hubs();
		compute_decrements(top_k);
        for (const auto &k: top_k) {
            fmt::print("[{} : {}], ", k, degrees[k]);
        }
        fmt::print("\n");
        fmt::print("k: {}\n", k);
		par_decrement();
		// compute connected components
		Afforest();
		par_compute_cc_sizes();
		if (gcc_size <= k) {
			break;
		}
		if (component_sizes.size() == 1) { // after hub removal, if graph is a single GCC, recur
			clear();
			continue;
		}

		par_compute_spokes();
//		fmt::print("perm: {}\n", perm);
//		fmt::print("gcc id: {}\n", gcc_id);
//		fmt::print("gcc size: {}\n", gcc_size);
//		print_bitmap();
//		fmt::print("degrees: {}\n", degrees);
		clear();
		n_iters++;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto slashburn_time = std::chrono::duration_cast<time_unit>(end - start);
	time = (uint64_t) slashburn_time.count();
}


void SlashBurn::clear() {
	spokes.clear();
	decrements.clear();
	component_sizes.clear();
	active_vertex_set.fill(0);
//	top_k.clear();
//	comp.fill(0);
}


void SlashBurn::par_decrement() {
	// convert to vector for parallel iteration
//	std::vector<std::pair<uint64_t, uint64_t>> v{decrements.begin(), decrements.end()};
	pvector<std::pair<uint64_t, uint64_t>> v(decrements.size());

	std::transform(
		dpl::execution::par,
		decrements.begin(),
		decrements.end(),
		v.begin(),
		[](auto &kv){ return kv;}
	);

	uint64_t n = v.size();
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < n; i++) {
		uint64_t vid = v[i].first;
		uint64_t decrement = v[i].second;
		if (degrees[vid] < decrement) { // todo: hacky; should be exactly 0 for last decrement
			degrees[vid] = 0;
		} else {
			degrees[vid] -= decrement;
		}

	}
}

void SlashBurn::compute_decrements(pvector<uint64_t> &vertices_to_decrement) {

#pragma omp parallel for schedule(dynamic, 16384)
	for (uint64_t i = 0; i < vertices_to_decrement.size(); i++) {
		uint64_t u = vertices_to_decrement[i];
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

		decrements[u] = n_incident_edges;

	}
//	fmt::print("decrements: {}\n", decrements);


}

void SlashBurn::print_bitmap() {
	fmt::print("[");
	for (uint64_t i = 0; i < num_nodes - 1; ++i) {
		if (bmap.get_bit(i)) { fmt::print("| "); }
		else { fmt::print("- "); }
	}
	if (bmap.get_bit(num_nodes - 1)) { fmt::print("|]\n"); }
	else { fmt::print("-]\n"); }
}

void SlashBurn::get_final_components(map_t &comps) {
// populate a map from component id to sets of vertex ids in that component
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < comp.size(); ++i) {
		uint64_t cid = comp[i];
		if (bmap.get_bit(i)) { continue; } // vertex has already been assigned a new id
		else {
			if (comps.contains(cid)) { comps[cid].insert(i); }
			else {
				comps.try_emplace_l(
					cid,
					[&i](map_t::value_type &val) { val.second.insert(i); }, // called only when cid already present in map
					set_t{i}); // otherwise, construct new vertex id set corresponding to cid
			}
		}
	}
}

void SlashBurn::place_final_comp_vec(map_t &comps) {
	pvector<std::vector<uint64_t>> comp_vec(comps.size());
	uint64_t comp_idx = 0;
	uint64_t comp_start_idx = spokes_end;

	for (auto &kv: comps) {
		uint64_t n_vs_in_comp = kv.second.size();
		comp_vec[comp_idx].reserve(comp_vec[comp_idx].size() + n_vs_in_comp);
		comp_vec[comp_idx].insert(
			comp_vec[comp_idx].end(),
			std::make_move_iterator(kv.second.begin()),
			std::make_move_iterator(kv.second.end()));
		// sort by ascending id within each spoke
//		std::sort(dpl::execution::par_unseq, comp_vec[comp_idx].begin(), comp_vec[comp_idx].end());
		++comp_idx;
		comp_start_idx -= n_vs_in_comp;
	}
	std::sort(dpl::execution::par_unseq,
	          comp_vec.begin(),
	          comp_vec.end(),
	          [](std::vector<uint64_t> s1, std::vector<uint64_t> s2) -> bool {
		          if (s1.size() == s2.size()) { return s1[0] < s2[0]; }
		          else { return s1.size() > s2.size(); }
	          });

	// compute a partial sum so that each spoke corresponds to an index into the final perm array
	std::vector<uint64_t> perm_start_idcs(comp_vec.size());
	std::transform(
		dpl::execution::par_unseq,
		comp_vec.begin(),
		comp_vec.end(),
		perm_start_idcs.begin(),
		[](std::vector<uint64_t> &l) -> uint64_t { return l.size(); }
	);
	std::exclusive_scan(
		dpl::execution::par_unseq,
		perm_start_idcs.begin(),
		perm_start_idcs.end(),
		perm_start_idcs.begin(),
		comp_start_idx,
		[](uint64_t l, uint64_t r) -> uint64_t { return l + r; }
	);
	// since spokes are sorted by descending size, omp schedule(static, 1) ensures that division
	// of work among threads is roughly equal
	// todo ^ is this a valid assumption?
#pragma omp parallel for schedule(static, 1)
	for (uint64_t i = 0; i < comp_vec.size(); ++i) {
		// as we iterate through spokes
		std::vector<uint64_t> &comp = comp_vec[i];
		uint64_t k = 0;
		for (uint64_t j = perm_start_idcs[i]; j < perm_start_idcs[i] + comp.size(); ++j) {
			perm[j] = comp[k];
			++k;
		}
	}
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
void SlashBurn::par_compute_spokes() {
// populate a map from component id to sets of vertex ids in that component
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < comp.size(); ++i) {
		uint64_t cid = comp[i];
		if (bmap.get_bit(i)) { continue; } // vertex has already been assigned a new id
		else if (cid == gcc_id) { continue; } // vertex is part of GCC, it will be assigned recursively
		else {
			// if (spokes.contains(cid)) { spokes[cid].insert(i); } not thread safe; .contains does not lock
			// if map contains key, lambda is called with the value_type  (under write lock protection),
			// and modify_if returns true. This is a non-const API and lambda is allowed to modify the mapped value
			bool cid_in_spokes = spokes.modify_if(
				cid,
				[&i](map_t::value_type &val) { val.second.insert(i); }
			);
			if (!cid_in_spokes) {
				spokes.try_emplace_l(
					cid,
					[&i](map_t::value_type &val) { val.second.insert(i); }, // called only when cid already present in map
					set_t{i}); // otherwise, construct new vertex id set corresponding to cid
			}
		}
	}

	// todo this step seems costly - could be done more efficiently
	// copy into a vector of vectors and sort
	pvector<std::vector<uint64_t>> spokes_vec(spokes.size());
	uint64_t spokes_idx = 0;
	uint64_t spoke_start_idx = spokes_end;

	for (auto &kv: spokes) {
		uint64_t n_vs_in_spoke = kv.second.size();
		spokes_vec[spokes_idx].reserve(spokes_vec[spokes_idx].size() + n_vs_in_spoke);
		spokes_vec[spokes_idx].insert(
			spokes_vec[spokes_idx].end(),
			std::make_move_iterator(kv.second.begin()),
			std::make_move_iterator(kv.second.end()));
		// sort by ascending id within each spoke
//		std::sort(dpl::execution::par_unseq, spokes_vec[spokes_idx].begin(), spokes_vec[spokes_idx].end());
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
	uint64_t n_spokes_to_remove = perm_start_idcs[perm_start_idcs.size() - 1] - spoke_start_idx + \
                                spokes_vec[spokes_vec.size() - 1].size();
	pvector<uint64_t> spokes_to_remove(n_spokes_to_remove);
	// since spokes are sorted by descending size, omp schedule(static, 1) ensures that division
	// of work among threads is roughly equal
	// todo ^ is this a valid assumption?
#pragma omp parallel for schedule(static, 1)
	for (uint64_t i = 0; i < spokes_vec.size(); ++i) {
		// as we iterate through spokes
		std::vector<uint64_t> &spoke = spokes_vec[i];
		uint64_t k = 0;
		for (uint64_t j = perm_start_idcs[i]; j < perm_start_idcs[i] + spoke.size(); ++j) {
			uint64_t l = j - spoke_start_idx; // index into spokes_to_remove vec
			spokes_to_remove[l] = spoke[k];
			perm[j] = spoke[k];
			bmap.set_bit_atomic(spoke[k]);
			++k;
		}
	}

	// compute the degree decrement after spoke removal
	decrements.clear();
	compute_decrements(spokes_to_remove);
	par_decrement();


//	remove_from_active_vertex_set(spokes_to_remove);

	spokes_end = spoke_start_idx;
}


/**
 * Given a vector of component id assignment, compute in parallel the number of vertices in each
 * component
 * Each open mp thread will compute a thread local absl::flat_hash_map of <component id -> num vertices>
 * Then use absl map.merge to merge all maps
 * i.e. map reduce
 * @param comp
 */
void SlashBurn::par_compute_cc_sizes() {
	std::map<int, absl::flat_hash_map<uint64_t, uint64_t>> thread_local_cc_counts;

	// init the thread local maps
	for (int i = 0; i < omp_get_max_threads(); ++i) { thread_local_cc_counts[i] = {}; }

	// each thread is assigned a static region of the component assignment vector
	// each thread builds a map between component id -> number of vertices in that component
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < comp.size(); ++i) {
		// skip over already assigned vertices
		if (bmap.get_bit(i)) { continue; }
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
		component_sizes.try_emplace_l(cid,
		                              [&n_vs_in_component](hash_t::value_type &val) { val.second += n_vs_in_component; },
		                              n_vs_in_component);
	}
	auto max_kv = find_max_value_pair(component_sizes);

	gcc_id = max_kv.first; // giant connected component ID
	gcc_size = max_kv.second;
	if (gcc_size <= k) {
		pvector<uint64_t> final_vids(component_sizes.size());
		uint64_t i = 0;
		map_t final_components;
		get_final_components(final_components);
		place_final_comp_vec(final_components);

//		for (auto &kv: component_sizes) {
//			final_vids[i] = kv.first;
//			i += 1;
//		}
//		fmt::print("final_vids: {}\n", final_vids);
//		ips4o::parallel::sort(
//			final_vids.begin(),
//			final_vids.end()
//		);
//		for (uint64_t i = 0; i < final_vids.size(); ++i) {
//			perm[hub_start] = final_vids[i];
//			hub_start++;
//		}
	}
}


std::pair<uint64_t, uint64_t> SlashBurn::find_max_value_pair(hash_t &x) {
	using value_t = typename hash_t::value_type;
	const auto compare = [](value_t const &p1, value_t const &p2) {
		return p1.second < p2.second;
	};
	return *std::max_element(dpl::execution::par_unseq, x.begin(), x.end(), compare);;
}

void SlashBurn::remove_k_hubs() {
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < k; ++i) {
		uint64_t hub_id = vids[i];
		top_k[i] = hub_id;
		perm[hub_start + i] = hub_id;
		bmap.set_bit_atomic(hub_id);
	}
//	remove_from_active_vertex_set(top_k);
	hub_start += k;

}


/*
 * iterate (in parallel) over the active vertex bitmap
 * each thread will iterate over an assigned section of the bitmap
 * for each active vertex seen, it will store that vertex' id in a local array
 * once all active vertices have been seen, merge them into a global array
 */
void SlashBurn::get_active_vertex_set() {

	pvector<pvector<uint64_t>> all_vs(omp_get_max_threads());
	uint64_t max_results_per_thread = num_nodes / omp_get_max_threads() + 1;
	for (int i = 0; i < omp_get_max_threads(); ++i) {
		all_vs[i].resize(max_results_per_thread);
	}
	uint64_t n_active_vertices = 0;
	// once all active vertices have been identified, the start, end points for each thread need to be stored
	// (they may not all be equal)
	pvector<uint64_t> bounds(omp_get_max_threads());
#pragma omp parallel
	{
		int t = omp_get_thread_num();
		pvector<uint64_t> &local_newq = all_vs[t];
		local_newq.resize(max_results_per_thread);
		uint64_t local_newq_count = 0;

		uint64_t start_bm_idx = t * max_results_per_thread;
		uint64_t end_bm_idx = (t + 1) * max_results_per_thread;
		end_bm_idx = (end_bm_idx > num_nodes) ? num_nodes : end_bm_idx;

		for (uint64_t i = start_bm_idx; i < end_bm_idx; ++i) {
			if (!bmap.get_bit(i)) {
				local_newq[local_newq_count] = i;
				++local_newq_count;
			}
		}
#pragma omp atomic update
		n_active_vertices += local_newq_count;

		bounds[t] = local_newq_count;
	}
	pvector<uint64_t> cumulative_bounds(bounds.size());
	std::exclusive_scan(
		bounds.begin(),
		bounds.end(),
		cumulative_bounds.begin(),
		0,
		[](uint64_t l, uint64_t r) -> uint64_t { return l + r; }
	);
	active_vertex_set.resize(n_active_vertices);
#pragma omp parallel
	{
		int t = omp_get_thread_num();
		uint64_t local_mx = bounds[t];
		uint64_t start = cumulative_bounds[t];
		for (int i = 0; i < local_mx; ++i) {
			active_vertex_set[start] = all_vs[t][i];
			++start;
		}
	}
}

void SlashBurn::init() {

	bmap.reset();
	degrees.resize(num_nodes);
	vids.resize(num_nodes);
	top_k.resize(k);
	perm.resize(num_nodes);
	comp.resize(num_nodes);
	perm.fill(-1);
	hub_start = 0;
	spokes_end = num_nodes;
#pragma omp parallel for
	for (uint64_t i = 0; i < g.num_nodes(); ++i) {
		vids[i] = i;
//		active_vertex_set.insert(i);
	}
	populate_degrees();
}

void SlashBurn::populate_degrees() {
#pragma omp parallel for schedule(static)
	for (uint64_t v = 0; v < num_nodes; ++v) {
		uint64_t d = g.out_degree(v) + g.in_degree(v);
		degrees[v] = d;
	}
}

void SlashBurn::sort_by_degrees() {
	ips4o::parallel::sort(
		vids.begin(),
		vids.end(),
		[&](const uint64_t &a, const uint64_t &b) {
			if (degrees[a] == degrees[b]) {
				return a < b;
			} else {
				return degrees[a] > degrees[b];
			}
		});
}


// Place nodes u and v in same component of lower component ID
void SlashBurn::Link(NodeID u, NodeID v) {
	NodeID p1 = comp[u];
	NodeID p2 = comp[v];
	while (p1 != p2) {
		NodeID high = p1 > p2 ? p1 : p2;
		NodeID low = p1 + (p2 - high);
		NodeID p_high = comp[high];
		// Was already 'low' or succeeded in writing 'low'
		if ((p_high == low) ||
		    (p_high == high && compare_and_swap(comp[high], high, low)))
			break;
		p1 = comp[comp[high]];
		p2 = comp[low];
	}
}


// Reduce depth of tree for each component to 1 by crawling up parents
void SlashBurn::Compress() {
#pragma omp parallel for schedule(dynamic, 16384)
	for (NodeID n = 0; n < g.num_nodes(); n++) {
		if (bmap.get_bit(n)) { continue; }
		while (comp[n] != comp[comp[n]]) {
			comp[n] = comp[comp[n]];
		}
	}
}


NodeID SlashBurn::SampleFrequentElement(int64_t num_samples = 1024) {
	std::unordered_map<NodeID, int> sample_counts(32);
	using kvp_type = std::unordered_map<NodeID, int>::value_type;
	// Sample elements from 'comp'
	// need to only sample from active vertices
	get_active_vertex_set();
    std::mt19937 gen;

    fmt::print("active_vertex_set: {}\n", active_vertex_set.size());
	std::uniform_int_distribution<NodeID> distribution(0, active_vertex_set.size() - 1);
	for (NodeID i = 0; i < num_samples; i++) {
		NodeID n = active_vertex_set[distribution(gen)];
		sample_counts[comp[n]]++;
	}

	// Find most frequent element in samples (estimate of most frequent overall)
	auto most_frequent = std::max_element(
		sample_counts.begin(), sample_counts.end(),
		[](const kvp_type &a, const kvp_type &b) { return a.second < b.second; });
    fmt::print("most_frequent: {}\n", most_frequent->first);
	float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
//	std::cout
//		<< "Skipping largest intermediate component (ID: " << most_frequent->first
//		<< ", approx. " << static_cast<int>(frac_of_graph * 100)
//		<< "% of the graph)" << std::endl;
	return most_frequent->first;
}


void SlashBurn::Afforest() {
//	pvector<NodeID> comp(g.num_nodes());

	// Initialize each node to a single-node self-pointing tree
#pragma omp parallel for
	for (NodeID n = 0; n < g.num_nodes(); n++) {
		comp[n] = n;
	}
	// Process a sparse sampled subgraph first for approximating components.
	// Sample by processing a fixed number of neighbors for each node (see paper)
	for (int r = 0; r < n_neighbour_rounds; ++r) {
#pragma omp parallel for schedule(dynamic, 16384)
		for (NodeID u = 0; u < g.num_nodes(); u++) {
			if (bmap.get_bit(u)) { continue; }
			for (NodeID v: g.out_neigh(u, r)) {
				if (bmap.get_bit(v)) { continue; }
				// Link at most one time if neighbor available at offset r
				Link(u, v);
				break;
			}
		}
		Compress();
	}

	// Sample 'comp' to find the most frequent element -- due to prior
	// compression, this value represents the largest intermediate component
	NodeID c = SampleFrequentElement(1024);
//	c = 0;

	// Final 'link' phase over remaining edges (excluding largest component)
	if (!g.directed()) {
#pragma omp parallel for schedule(dynamic, 16384)
		for (NodeID u = 0; u < g.num_nodes(); u++) {
			// skip vertices already assigned
			if (bmap.get_bit(u)) { continue; }
			// Skip processing nodes in the largest component
			if (comp[u] == c) { continue; }
			// Skip over part of neighborhood (determined by neighbor_rounds)
			for (NodeID v: g.out_neigh(u, n_neighbour_rounds)) {
				if (bmap.get_bit(v)) { continue; }
				Link(u, v);
			}
		}
	} else {

#pragma omp parallel for schedule(dynamic, 16384)
		for (NodeID u = 0; u < g.num_nodes(); u++) {
			if (bmap.get_bit(u)) { continue; }
			if (comp[u] == c)
				continue;
			for (NodeID v: g.out_neigh(u, n_neighbour_rounds)) {
				if (bmap.get_bit(v)) { continue; }
				Link(u, v);
			}
			// To support directed graphs, process reverse graph completely
			for (NodeID v: g.in_neigh(u)) {
				if (bmap.get_bit(v)) { continue; }
				Link(u, v);
			}
		}
	}
	// Finally, 'compress' for final convergence
	Compress();
//	return comp;
}


void SlashBurn::PrintCompStats() {
	std::cout << std::endl;
	std::unordered_map<NodeID, NodeID> count;
	for (NodeID comp_i: comp)
		count[comp_i] += 1;
	int k = 5;
	std::vector<std::pair<NodeID, NodeID>> count_vector;
	count_vector.reserve(count.size());
	for (auto kvp: count)
		count_vector.push_back(kvp);
	std::vector<std::pair<NodeID, NodeID>> top_k = TopK(count_vector, k);
	k = std::min(k, static_cast<int>(top_k.size()));
	std::cout << k << " biggest clusters" << std::endl;
	for (auto kvp: top_k)
		std::cout << kvp.second << ":" << kvp.first << std::endl;
	std::cout << "There are " << count.size() << " components" << std::endl;
}


// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
bool SlashBurn::CCVerifier() {
	std::unordered_map<NodeID, NodeID> label_to_source;
	for (NodeID n: g.vertices())
		label_to_source[comp[n]] = n;
	Bitmap visited(g.num_nodes());
	visited.reset();
	std::vector<NodeID> frontier;
	frontier.reserve(g.num_nodes());
	for (auto label_source_pair: label_to_source) {
		NodeID curr_label = label_source_pair.first;
		NodeID source = label_source_pair.second;
		frontier.clear();
		frontier.push_back(source);
		visited.set_bit(source);
		for (auto it = frontier.begin(); it != frontier.end(); it++) {
			NodeID u = *it;
			for (NodeID v: g.out_neigh(u)) {
				if (comp[v] != curr_label)
					return false;
				if (!visited.get_bit(v)) {
					visited.set_bit(v);
					frontier.push_back(v);
				}
			}
			if (g.directed()) {
				for (NodeID v: g.in_neigh(u)) {
					if (comp[v] != curr_label)
						return false;
					if (!visited.get_bit(v)) {
						visited.set_bit(v);
						frontier.push_back(v);
					}
				}
			}
		}
	}
	for (NodeID n = 0; n < g.num_nodes(); n++)
		if (!visited.get_bit(n))
			return false;
	return true;
}

void SlashBurn::write_permutation(std::string path) {
	std::ofstream outfile(path);
	outfile << fmt::format("{}\n", num_nodes);
	outfile << fmt::format("{}\n", g.num_edges());

	// create the map and sort by original id
	pvector<std::pair<uint64_t, uint64_t>> p(num_nodes);
#pragma omp parallel for schedule(static)
	for (uint64_t i = 0; i < num_nodes; ++i) {
		p[i] = {perm[i], i};
	}

	std::sort(dpl::execution::par_unseq, p.begin(), p.end());

	for (uint64_t i = 0; i < num_nodes; ++i) {
		outfile << fmt::format("{} {}\n", p[i].first, p[i].second);
	}

	outfile.close();
}