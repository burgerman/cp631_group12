#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include "mpi.h"

#define FOREST_SIZE 6
#define ITERATIONS 3
#define STATE_EMPTY 0
#define STATE_TREE 1
#define STATE_BURNING 2
#define STATE_BURNT 3
#define OUTPUT_NAME "FOREST_FILE"


/* Function Declarations */
void initialize_forest(int forest_rank,  int **local_forest, int local_forest_size, int **simulated_forest);
void simulate_forest(int **local_forest, MPI_Comm forest_comm, int local_forest_size, int **simulated_forest);
int predict_if_tree_burn(int row, int col, int local_forest_size, int **local_forest, int **neighboring_states);
void output_forest_to_file(int **simulated_forest, MPI_Comm forest_comm, int forest_rank, int coordinates[], int dim, int local_forest_size);
void update_local_forest(int local_forest_size, int **local_forest, int **simulated_forest);


int main(int argc, char *argv[]) {
    int i, j, my_rank, forest_rank, p, q;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    q = (int) sqrt((double) p);
    int local_forest_size = FOREST_SIZE/q;
    int **local_forest = (int **)malloc(local_forest_size * sizeof(int *));
    for(i=0; i<local_forest_size; i++) {
        local_forest[i] = (int *)malloc(local_forest_size * sizeof(int));
    }

    int **simulated_forest = (int **)malloc(local_forest_size * sizeof(int *));
    for(i=0; i<local_forest_size; i++) {
        simulated_forest[i] = (int *)malloc(local_forest_size * sizeof(int));
    }

    int dimensions[2];
    dimensions[0] = dimensions[1] = q;
    int periods[2];
    periods[0] = periods[1] = 0;
    MPI_Comm  forest_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &forest_comm);
    MPI_Comm_rank(forest_comm, &forest_rank);
    int coordinates[2];
    MPI_Cart_coords(forest_comm, forest_rank, 2, coordinates);
    MPI_Cart_rank(forest_comm, coordinates, &forest_rank);
    // ITERATIONS: Assume how many forests there are that we need to simulate
    for (i=0; i<ITERATIONS; i++) {
        // ITERATIONS: Assume how many times we need to simulate for each forest
        initialize_forest(forest_rank, local_forest, local_forest_size, simulated_forest);
        for (j=0; j<ITERATIONS; j++) {
            // Output the current forest to the file
            output_forest_to_file(simulated_forest, forest_comm, coordinates);
            // Simulate how the forest will be like
            simulate_forest(local_forest, forest_comm, local_forest_size, simulated_forest);
            // Update the local forest after the simulation
            update_local_forest(local_forest_size, local_forest, simulated_forest);
            MPI_Barrier(forest_comm);
        }
        MPI_Barrier(forest_comm);
    }

    for(i=0; i<local_forest_size; i++) {
        free(local_forest[i]);
        free(simulated_forest[i]);
    }
    free(local_forest);
    free(simulated_forest);
    MPI_Finalize();
    return 0;
}



void initialize_forest(int forest_rank,  int **local_forest, int local_forest_size, int **simulated_forest) {
    int i, j;
    srand(forest_rank+time(NULL));
    for (i=0; i<local_forest_size; i++) {
        for(j=0; j<local_forest_size; j++) {
            local_forest[i][j] = rand() % 4;
            simulated_forest[i][j] = local_forest[i][j];
        }
    }
}



void simulate_forest(int **local_forest, MPI_Comm forest_comm, int local_forest_size, int **simulated_forest) {
    int i, j, up, down, left, right;

    MPI_Cart_shift(forest_comm, 0, 1, &up, &down);
    MPI_Cart_shift(forest_comm, 1, 1, &left, &right);

    /*
     * neighboring_states[0] - store up neighboring elements
     * neighboring_states[1] - store down neighboring elements
     * neighboring_states[2] - store left neighboring elements
     * neighboring_states[3] - store right neighboring elements
     */
    int **neighboring_states = (int **)malloc(4 * sizeof(int *));
    for(i=0; i<4; i++) {
        neighboring_states[i] = (int *)malloc(local_forest_size * sizeof(int));
    }
    if(up!= MPI_PROC_NULL && down != MPI_PROC_NULL) {
        MPI_Sendrecv(local_forest[local_forest_size-1], local_forest_size, MPI_INT, down, 0, neighboring_states[0], local_forest_size, MPI_INT, up, 0, forest_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(local_forest[0], local_forest_size, MPI_INT, up, 0, neighboring_states[1], local_forest_size, MPI_INT, down, 0, forest_comm, MPI_STATUS_IGNORE);
    }
    else {
        if (up == MPI_PROC_NULL) {
            for(j=0; j<local_forest_size; j++) {
                neighboring_states[0][j] = STATE_EMPTY;
            }
            MPI_Sendrecv(local_forest[local_forest_size-1], local_forest_size, MPI_INT, down, 0, neighboring_states[1], local_forest_size, MPI_INT, down, 0, forest_comm, MPI_STATUS_IGNORE);
        } else if (down == MPI_PROC_NULL) {
            for(j=0; j<local_forest_size; j++) {
                neighboring_states[1][j] = STATE_EMPTY;
            }
            MPI_Sendrecv(local_forest[0], local_forest_size, MPI_INT, up, 0, neighboring_states[0], local_forest_size, MPI_INT, up, 0, forest_comm, MPI_STATUS_IGNORE);
        }
    }

    int left_col[local_forest_size], right_col[local_forest_size];
    for(i=0; i<local_forest_size; i++) {
        left_col[i] = local_forest[i][0];
        right_col[i] = local_forest[i][local_forest_size-1];
    }

    if (left!= MPI_PROC_NULL && right != MPI_PROC_NULL) {
        MPI_Sendrecv(right_col, local_forest_size, MPI_INT, right, 0, neighboring_states[2], local_forest_size, MPI_INT, left, 0, forest_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(left_col, local_forest_size, MPI_INT, left, 0, neighboring_states[3], local_forest_size, MPI_INT, right, 0, forest_comm, MPI_STATUS_IGNORE);
    }
    else {
        if(left == MPI_PROC_NULL) {
            for(j=0; j<local_forest_size; j++) {
                neighboring_states[2][j] = STATE_EMPTY;
            }
            MPI_Sendrecv(right_col, local_forest_size, MPI_INT, right, 0, neighboring_states[3], local_forest_size, MPI_INT, right, 0, forest_comm, MPI_STATUS_IGNORE);
        } else if (right == MPI_PROC_NULL) {
            for(j=0; j<local_forest_size; j++) {
                neighboring_states[3][j] = STATE_EMPTY;
            }
            MPI_Sendrecv(left_col, local_forest_size, MPI_INT, left, 0, neighboring_states[2], local_forest_size, MPI_INT, left, 0, forest_comm, MPI_STATUS_IGNORE);
        }
    }

    #pragma omp parallel for shared(local_forest, simulated_forest, neighboring_states, local_forest_size, up, down, left, right) collapse(2)
    for (i = 0; i < local_forest_size; i++) {
        for (j = 0; j < local_forest_size; j++) {
            if (local_forest[i][j] == STATE_BURNING) {
                simulated_forest[i][j] = STATE_BURNT;
            } else if ((local_forest[i][j] == STATE_EMPTY || local_forest[i][j] == STATE_BURNT)) {
                simulated_forest[i][j] = local_forest[i][j];
            } else {
                simulated_forest[i][j] = predict_if_tree_burn(i, j, local_forest_size,
                                                              local_forest, neighboring_states);
            }
        }
    }

    for(i=0; i<4; i++) {
        free(neighboring_states[i]);
    }
    free(neighboring_states);
}



int predict_if_tree_burn(int row, int col, int local_forest_size, int **local_forest, int **neighboring_states) {
    int up_entry, down_entry, left_entry, right_entry;

    if(row == 0) {
        up_entry = neighboring_states[0][col];
    } else {
        up_entry = local_forest[row-1][col];
    }

    if(row == local_forest_size-1) {
        down_entry = neighboring_states[1][col];
    } else {
        down_entry = local_forest[row+1][col];
    }

    if(col == 0) {
        left_entry = neighboring_states[2][row];
    } else {
        left_entry = local_forest[row][col-1];
    }

    if(col == local_forest_size -1) {
        right_entry = neighboring_states[3][row];
    } else {
        right_entry = local_forest[row][col+1];
    }

    if(up_entry == STATE_BURNING || down_entry == STATE_BURNING || left_entry == STATE_BURNING || right_entry == STATE_BURNING) {
        return STATE_BURNING;
    } else {
        return STATE_TREE;
    }
}



void output_forest_to_file(int **simulated_forest, MPI_Comm forest_comm, int coordinates[]) {

}

void update_local_forest(int local_forest_size, int **local_forest, int **simulated_forest) {
    int i, j;
    for(i=0; i<local_forest_size; i++) {
        for(j=0; j<local_forest_size; j++) {
            local_forest[i][j] = simulated_forest[i][j];
        }
    }
}
