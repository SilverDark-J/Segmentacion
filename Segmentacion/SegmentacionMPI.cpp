#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image-mpi.h"

#include <iostream>
#include <exception>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        float sigma = 0.5;
        float k = 500;
        int min_size = 20;

        const char* input_filename = "../image_data/cat.pnm";
        const char* output_filename = "../image_results/cat_mpi.ppm";

        // Tiempo de inicio
        double start_time = MPI_Wtime();

        image<rgb>* input = nullptr;
        if (rank == 0) {
            printf("Loading input image...\n");
            input = loadPPM(input_filename);
            if (!input) {
                std::cerr << "Error: failed to load image " << input_filename << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        int num_ccs;
        image<rgb>* output = segment_image_mpi((rank == 0) ? input : nullptr, sigma, k, min_size, &num_ccs, rank, size);

        if (rank == 0) {
            if (!output) {
                std::cerr << "Error: segmentation returned nullptr.\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            printf("Saving segmented image...\n");
            savePPM(output, output_filename);
            printf("Got %d components\n", num_ccs);
            delete input;
            delete output;
        }

        // Tiempo de fin
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        // Solo el proceso 0 imprime el tiempo
        if (rank == 0) {
            printf("Tiempo total de ejecución (MPI): %.6f segundos\n", elapsed_time);
        }

        MPI_Finalize();
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Unhandled C++ exception: " << ex.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 99);
        return 99;
    }
    catch (...) {
        std::cerr << "Unknown fatal error.\n";
        MPI_Abort(MPI_COMM_WORLD, 100);
        return 100;
    }
}
