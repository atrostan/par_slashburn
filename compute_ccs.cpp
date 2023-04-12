
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
#include "cc.h"

// The hooking condition (comp_u < comp_v) may not coincide with the edge's
// direction, so we use a min-max swap such that lower component IDs propagate
// independent of the edge's direction.
pvector<NodeID> ShiloachVishkin(const Graph &g) {
	pvector<NodeID> comp(g.num_nodes());
#pragma omp parallel for
	for (NodeID n=0; n < g.num_nodes(); n++)
		comp[n] = n;
	bool change = true;
	int num_iter = 0;
	while (change) {
		change = false;
		num_iter++;
#pragma omp parallel for
		for (NodeID u=0; u < g.num_nodes(); u++) {
			for (NodeID v : g.out_neigh(u)) {
				NodeID comp_u = comp[u];
				NodeID comp_v = comp[v];
				if (comp_u == comp_v) continue;
				// Hooking condition so lower component ID wins independent of direction
				NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
				NodeID low_comp = comp_u + (comp_v - high_comp);
				if (high_comp == comp[high_comp]) {
					change = true;
					comp[high_comp] = low_comp;
				}
			}
		}
#pragma omp parallel for
		for (NodeID n=0; n < g.num_nodes(); n++) {
			while (comp[n] != comp[comp[n]]) {
				comp[n] = comp[comp[n]];
			}
		}
	}
	std::cout << "Shiloach-Vishkin took " << num_iter << " iterations" << std::endl;
	return comp;
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

int main(int argc, char *argv[])
{
    CLApp cli(argc, argv, "connected-components-afforest");
    if (!cli.ParseArgs())
        return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
//	pvector<NodeID> labels = Afforest(g, 2);
	pvector<NodeID> labels = ShiloachVishkin(g);


	std::vector<uint32_t> components(g.num_nodes());

	#pragma omp parallel for 
	for (uint32_t i = 0; i < g.num_nodes(); ++i) {
		components[i] = labels[i];
	}

	write_vector_as_bin(cli.output_path(), components);

	// translate to vector of uint32_t to write to file 
		// std::ofstream output("./tmp.ccs");
		// for (const auto & l: labels) {
			// output << l << "\n";
		// }
		// output.close();

    return 0;
}