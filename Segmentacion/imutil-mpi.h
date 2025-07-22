#ifndef IMUTIL_MPI_H
#define IMUTIL_MPI_H

#include "image.h"
#include "misc.h"
#include <mpi.h>

/* compute minimum and maximum value in an image */
template <class T>
void min_max(image<T>* im, T* global_min, T* global_max) {
    int width = im->width();
    int height = im->height();

    T local_min = imRef(im, 0, 0);
    T local_max = imRef(im, 0, 0);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_start = (rank * height) / size;
    int local_end = ((rank + 1) * height) / size;

    for (int y = local_start; y < local_end; y++) {
        for (int x = 0; x < width; x++) {
            T val = imRef(im, x, y);
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }
    }

    MPI_Allreduce(&local_min, global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

/* threshold image */
template <class T>
image<uchar>* threshold(image<T>* src, int t) {
    int width = src->width();
    int height = src->height();
    image<uchar>* dst = new image<uchar>(width, height);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_start = (rank * height) / size;
    int local_end = ((rank + 1) * height) / size;

    for (int y = local_start; y < local_end; y++) {
        for (int x = 0; x < width; x++) {
            imRef(dst, x, y) = (imRef(src, x, y) >= t);
        }
    }

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        dst->data, (width * height) / size, MPI_UNSIGNED_CHAR,
        MPI_COMM_WORLD);

    return dst;
}

#endif
