#ifndef FILTER_MPI_H
#define FILTER_MPI_H

#include <vector>
#include <cmath>
#include <mpi.h>
#include "image.h"
#include "misc.h"
#include "convolve.h"
#include "imconv.h"

#define WIDTH 4.0

static void normalize(std::vector<float>& mask) {
    int len = mask.size();
    float sum = 0;
    for (int i = 1; i < len; i++) {
        sum += fabs(mask[i]);
    }
    sum = 2 * sum + fabs(mask[0]);
    for (int i = 0; i < len; i++) {
        mask[i] /= sum;
    }
}

#define MAKE_FILTER(name, fun)                                \
static std::vector<float> make_ ## name (float sigma) {       \
  sigma = std::max(sigma, 0.01F);			      \
  int len = (int)ceil(sigma * WIDTH) + 1;                     \
  std::vector<float> mask(len);                               \
  for (int i = 0; i < len; i++) {                             \
    mask[i] = fun;                                            \
  }                                                           \
  return mask;                                                \
}

MAKE_FILTER(fgauss, exp(-0.5 * square(i / sigma)));

static image<float>* smooth(image<float>* src, float sigma) {
    std::vector<float> mask = make_fgauss(sigma);
    normalize(mask);

    image<float>* tmp = new image<float>(src->height(), src->width(), false);
    image<float>* dst = new image<float>(src->width(), src->height(), false);
    convolve_even(src, tmp, mask);
    convolve_even(tmp, dst, mask);

    delete tmp;
    return dst;
}

image<float>* smooth(image<uchar>* src, float sigma) {
    image<float>* tmp = imageUCHARtoFLOAT(src);
    image<float>* dst = smooth(tmp, sigma);
    delete tmp;
    return dst;
}

static image<float>* laplacian(image<float>* src) {
    int width = src->width();
    int height = src->height();
    image<float>* dst = new image<float>(width, height);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = height / size;
    int remainder = height % size;

    int start_row = rank * rows_per_proc + std::min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    std::vector<float> local_data(local_rows * width, 0.0f);
    for (int y = 0; y < local_rows; y++) {
        int global_y = start_row + y;
        if (global_y <= 0 || global_y >= height - 1) continue;
        for (int x = 1; x < width - 1; x++) {
            float d2x = imRef(src, x - 1, global_y) + imRef(src, x + 1, global_y) - 2 * imRef(src, x, global_y);
            float d2y = imRef(src, x, global_y - 1) + imRef(src, x, global_y + 1) - 2 * imRef(src, x, global_y);
            local_data[y * width + x] = d2x + d2y;
        }
    }

    std::vector<int> recvcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rpp = height / size + (i < remainder ? 1 : 0);
        recvcounts[i] = rpp * width;
        displs[i] = offset;
        offset += recvcounts[i];
    }

    std::vector<float> all_data(width * height);
    MPI_Gatherv(local_data.data(), local_rows * width, MPI_FLOAT,
        all_data.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                imRef(dst, x, y) = all_data[y * width + x];
    }

    return dst;
}

#endif
