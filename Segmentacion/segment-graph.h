#ifndef SEGMENT_GRAPH
#define SEGMENT_GRAPH

#include <algorithm>
#include <cmath>
#include "disjoint-set.h"

// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct {
    float w;
    int a, b;
} edge;

bool operator<(const edge& a, const edge& b) {
    return a.w < b.w;
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
universe* segment_graph(int num_vertices, int num_edges, edge* edges,
    float c) {
    // sort edges by weight
    std::sort(edges, edges + num_edges);

    // make a disjoint-set forest
    universe* u = new universe(num_vertices);

    // init thresholds
    float* threshold = new float[num_vertices];
    for (int i = 0; i < num_vertices; i++)
        threshold[i] = THRESHOLD(1, c);

    // for each edge, in non-decreasing weight order...
    for (int i = 0; i < num_edges; i++) {
        edge* pedge = &edges[i];

        // components conected by this edge
        int a = u->find(pedge->a);
        int b = u->find(pedge->b);
        if (a != b) {
            if ((pedge->w <= threshold[a]) &&
                (pedge->w <= threshold[b])) {
                u->join(a, b);
                a = u->find(a);
                threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
            }
        }
    }

    // free up
    delete[] threshold;
    return u;
}

#endif
