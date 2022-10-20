#include <iostream>
//#include "command_line.h"
//#include "benchmark.h"
#include "cc.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include "utilities.h"
#include "par_utils.h"
#include "ips4o.hpp"
#include "SlashBurn.h"
#include <boost/filesystem/path.hpp>
#include "sql.h"

bool degree_compare(const std::pair<uint64_t, uint64_t> &a, const std::pair<uint64_t, uint64_t> &b) {
	if (a.second == b.second) {
		return a.first < b.first;
	}
	return a.second > b.second;
}

int main(int argc, char *argv[]) {
	CLApp cli(argc, argv, "connected-components-afforest");
	if (!cli.ParseArgs())
		return -1;
	Builder b(cli);
	fmt::print("cli.out_filename(): {}\n", cli.out_filename());
	Graph g = b.MakeGraph();
	int n_neighbour_rounds = 2;
	float percent = cli.percent();
  // TODO add k as input arg
	Bitmap bmap(g.num_nodes());
	SlashBurn sb = SlashBurn(g, n_neighbour_rounds, percent, bmap);
	sb.write_permutation(cli.out_filename());

	boost::filesystem::path p(cli.filename());
	boost::filesystem::path dir = p.parent_path();
	std::string graph_name = dir.filename().string();
	std::string sqlite_db_path = cli.db_filename();
	single_val_set_int(sqlite_db_path, "par_slashburn", "preproc", graph_name, sb.time);
	single_val_set_int(sqlite_db_path, "par_sb_k", "statistics", graph_name, sb.k);
	single_val_set_int(sqlite_db_path, "par_sb_n_iters", "statistics", graph_name, sb.n_iters);
	return 0;
}