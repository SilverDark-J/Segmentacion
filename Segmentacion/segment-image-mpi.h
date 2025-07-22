#ifndef SEGMENT_IMAGE_MPI
#define SEGMENT_IMAGE_MPI

#include <cstdlib>
#include <vector>
#include <mpi.h>
#include "image.h"
#include "misc.h"
#include "filter-mpi.h"
#include "segment-graph.h"

#include <iostream>

// random color
rgb random_rgb() {
    rgb c;
    c.r = (uchar)(rand() % 256);
    c.g = (uchar)(rand() % 256);
    c.b = (uchar)(rand() % 256);
    return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float>* r, image<float>* g, image<float>* b,
    int x1, int y1, int x2, int y2) {
    return sqrt(square(imRef(r, x1, y1) - imRef(r, x2, y2)) +
        square(imRef(g, x1, y1) - imRef(g, x2, y2)) +
        square(imRef(b, x1, y1) - imRef(b, x2, y2)));
}

MPI_Datatype create_mpi_edge_type() {
    MPI_Datatype mpi_edge_type;
    int lengths[3] = { 1, 1, 1 };
    MPI_Aint displacements[3];
    displacements[0] = offsetof(edge, w);
    displacements[1] = offsetof(edge, a);
    displacements[2] = offsetof(edge, b);
    MPI_Datatype types[3] = { MPI_FLOAT, MPI_INT, MPI_INT };
    MPI_Type_create_struct(3, lengths, displacements, types, &mpi_edge_type);
    MPI_Type_commit(&mpi_edge_type);
    return mpi_edge_type;
}

// Segmentación de imagen usando MPI
image<rgb>* segment_image_mpi(image<rgb>* im, float sigma, float c, int min_size,
    int* num_ccs, int rank, int size) {

    int width = 0, height = 0;

    if (rank == 0 && im != nullptr) {
        width = im->width();
        height = im->height();
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && height % size != 0)
        std::cout << "Advertencia: la imagen no se divide exactamente entre procesos.\n";

    image<float>* smooth_r = nullptr;
    image<float>* smooth_g = nullptr;
    image<float>* smooth_b = nullptr;

    if (rank == 0) {
        image<float>* r = new image<float>(width, height);
        image<float>* g = new image<float>(width, height);
        image<float>* b = new image<float>(width, height);

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                imRef(r, x, y) = imRef(im, x, y).r;
                imRef(g, x, y) = imRef(im, x, y).g;
                imRef(b, x, y) = imRef(im, x, y).b;
            }

        smooth_r = smooth(r, sigma);
        smooth_g = smooth(g, sigma);
        smooth_b = smooth(b, sigma);
        delete r; delete g; delete b;
    }
    else {
        smooth_r = new image<float>(width, height);
        smooth_g = new image<float>(width, height);
        smooth_b = new image<float>(width, height);
    }

    MPI_Bcast(smooth_r->data, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(smooth_g->data, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(smooth_b->data, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows_per_proc = height / size;
    int start_y = rank * rows_per_proc;
    int end_y = (rank == size - 1) ? height : start_y + rows_per_proc;

    std::vector<edge> local_edges;
    local_edges.reserve((end_y - start_y) * width * 4); // Evitar realocaciones

    for (int y = start_y; y < end_y; y++) {
        for (int x = 0; x < width; x++) {
            if (x < width - 1)
                local_edges.push_back({ diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y), y * width + x, y * width + (x + 1) });
            if (y < height - 1)
                local_edges.push_back({ diff(smooth_r, smooth_g, smooth_b, x, y, x, y + 1), y * width + x, (y + 1) * width + x });
            if ((x < width - 1) && (y < height - 1))
                local_edges.push_back({ diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y + 1), y * width + x, (y + 1) * width + (x + 1) });
            if ((x < width - 1) && (y > 0))
                local_edges.push_back({ diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y - 1), y * width + x, (y - 1) * width + (x + 1) });
        }
    }

    int local_count = static_cast<int>(local_edges.size());

    MPI_Datatype mpi_edge_type = create_mpi_edge_type();

    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    edge* all_edges = nullptr;
    int total_edges = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            displs[i] = total_edges;
            total_edges += recv_counts[i];
        }
        all_edges = new edge[total_edges];
    }

    MPI_Gatherv(local_edges.data(), local_count, mpi_edge_type,
        all_edges, recv_counts.data(), displs.data(), mpi_edge_type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&mpi_edge_type);

    image<rgb>* output = nullptr;

    if (rank == 0) {
        delete smooth_r; delete smooth_g; delete smooth_b;

        universe* u = segment_graph(width * height, total_edges, all_edges, c);

        for (int i = 0; i < total_edges; i++) {
            int a = u->find(all_edges[i].a);
            int b = u->find(all_edges[i].b);
            if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
                u->join(a, b);
        }

        *num_ccs = u->num_sets();
        output = new image<rgb>(width, height);
        rgb* colors = new rgb[width * height];
        for (int i = 0; i < width * height; i++)
            colors[i] = random_rgb();

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                int comp = u->find(y * width + x);
                imRef(output, x, y) = colors[comp];
            }

        delete[] colors;
        delete u;
        delete[] all_edges;
    }
    else {
        delete smooth_r; delete smooth_g; delete smooth_b;
    }

    return output;
}

#endif
