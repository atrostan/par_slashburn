//
// Created by atrostan on 13/10/22.
//

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
//#include "timer.h"

#ifndef GAPBS_SB_CC_H
#define GAPBS_SB_CC_H

void Link(NodeID u, NodeID v, pvector<NodeID> &comp);

void Compress(const Graph &g, pvector<NodeID> &comp);

NodeID SampleFrequentElement(const pvector<NodeID> &comp,
                             int64_t num_samples);

pvector<NodeID> Afforest(const Graph &g, int32_t neighbor_rounds);

void PrintCompStats(const Graph &g, const pvector<NodeID> &comp);

bool CCVerifier(const Graph &g, const pvector<NodeID> &comp);

#endif //GAPBS_SB_CC_H
