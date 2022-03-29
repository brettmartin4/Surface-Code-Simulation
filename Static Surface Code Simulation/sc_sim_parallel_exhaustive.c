/*	2d Lt Brett Martin
 *	Advisor: Dr. Laurence Merkle
 *	CSCE 656 - Parallel and Distributed Processing Algorithms
 *	Term project: parallel surface code simulation
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>


int main (int argc, char** argv) {

    // MPI vars (NOTE: Per the Navy DSRC intro guide, the standard queue has a maximum cores per job of 8,168 cores)
    int iproc, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Execution time variables:
    double exec_time = 0.0;
    clock_t begin;
    if (iproc == 0) {
        begin = clock();
    }

	//Inputs:
	int depth = 3;		// Or code distance
	float p_x_error = 0.05;
	float p_z_error = 0.05;

    int max_char[] = {128, 640, 328};
    int i_max_char = max_char[(depth%3)];

	int num_data_qubits = depth * depth;
	int data_qubit_x_error[ depth + 2][ depth + 2 ];
	int data_qubit_z_error[ depth + 2  ][ depth + 2 ];
	int ancilla_qubit_value[ depth + 1 ][ depth + 1 ];

	// Parallel block decomposition variables
	int total_num_iter = pow(4, num_data_qubits);			// Total number of outer loop iterations
	int block_size = floor(total_num_iter/nproc);		// Number of iterations per process (block)

	if (total_num_iter%nproc > 0) { block_size += 1; }	// Add 1 if blocks don't divide evenly

	int iter_first = iproc * block_size;
	int iter_last = iter_first + block_size;

    MPI_Status status;
    MPI_File fh;

    char buf[i_max_char];
    MPI_Offset offset = (MPI_Offset) (iproc * block_size * i_max_char);

    //Output:
    //list of records in which each possible combination of data qubit errors is associated with the resulting ancilla qubit values (may be probabilistic), along with the probability of the combination of data qubit errors (and measurement errors?)
    MPI_File_open(MPI_COMM_WORLD, "testfile.csv", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);

    printf("(%d) Writing to file...\n", iproc);
    printf("\nRank: (%d), Offset: %d\n", iproc, (int)offset);
    printf("Process %d range: %d to %d \n", iproc, iter_first, iter_last);


	for ( int i = iter_first; i < iter_last; i++ ) {

        int errors = i;
        double probability = 1.0;

        // initialize data_qubit_x_error and data_qubit_z_error to be full of FALSE (or 0) values
        memset(data_qubit_x_error, 0, sizeof(data_qubit_x_error));
        memset(data_qubit_z_error, 0, sizeof(data_qubit_z_error));
        memset(ancilla_qubit_value, 0, sizeof(ancilla_qubit_value));

        // Set data qubit errors and calculate probabilities
        for ( int j = 0; j < num_data_qubits; j++ ) {
            int this_error = errors % 2;
            data_qubit_x_error[ ( j / depth ) + 1][ ( j % depth ) + 1 ] = this_error;
            probability *= this_error ? p_x_error : (1.0 - p_x_error);
            errors /= 2;

            this_error = errors % 2;
            data_qubit_z_error[ ( j / depth ) + 1][ ( j % depth ) + 1 ] = this_error;
            errors /= 2;
            probability *= this_error ? p_z_error : (1.0 - p_z_error);
        }


        for ( int j = 0; j < (depth + 1 ) * (depth + 1 ) - 1; j++ ) {
            if ((j/(depth+1))%2 == 0) {
                ancilla_qubit_value[ j / (depth+1) ][ j % (depth+1) ] =
                        data_qubit_x_error[   j / (depth+1)       ][   j % (depth+1)       ]
                        ^ data_qubit_x_error[   j / (depth+1)       ][ ( j % (depth+1) ) + 1 ]
                        ^ data_qubit_x_error[ ( j / (depth+1) ) + 1 ][   j % (depth+1)       ]
                        ^ data_qubit_x_error[ ( j / (depth+1) ) + 1 ][ ( j % (depth+1) ) + 1 ];
                j++;
                ancilla_qubit_value[ j / (depth+1) ][ j % (depth+1) ] =
                        data_qubit_z_error[   j / (depth+1)       ][   j % (depth+1)       ]
                        ^ data_qubit_z_error[   j / (depth+1)       ][ ( j % (depth+1) ) + 1 ]
                        ^ data_qubit_z_error[ ( j / (depth+1) ) + 1 ][   j % (depth+1)       ]
                        ^ data_qubit_z_error[ ( j / (depth+1) ) + 1 ][ ( j % (depth+1) ) + 1 ];
            } else {
                ancilla_qubit_value[ j / (depth+1) ][ j % (depth+1) ] =
                        data_qubit_z_error[   j / (depth+1)       ][   j % (depth+1)       ]
                        ^ data_qubit_z_error[   j / (depth+1)       ][ ( j % (depth+1) ) + 1 ]
                        ^ data_qubit_z_error[ ( j / (depth+1) ) + 1 ][   j % (depth+1)       ]
                        ^ data_qubit_z_error[ ( j / (depth+1) ) + 1 ][ ( j % (depth+1) ) + 1 ];
                j++;
                ancilla_qubit_value[ j / (depth+1) ][ j % (depth+1) ] =
                        data_qubit_x_error[   j / (depth+1)       ][   j % (depth+1)       ]
                        ^ data_qubit_x_error[   j / (depth+1)       ][ ( j % (depth+1) ) + 1 ]
                        ^ data_qubit_x_error[ ( j / (depth+1) ) + 1 ][   j % (depth+1)       ]
                        ^ data_qubit_x_error[ ( j / (depth+1) ) + 1 ][ ( j % (depth+1) ) + 1 ];
            }
        }

        char label_list[i_max_char];
        strcpy(label_list, "\n");
        char anc_name[3];

        // Output the ancilla qubit values in proper format
        int ancilla_value;
        for (int k=1; k < depth; k++) {
            if (k%2 == 0) {
                ancilla_value = (ancilla_qubit_value[depth][k] == 1) ? -1 : 1;
                sprintf(anc_name, "%d,", ancilla_value);
                strcat(label_list, anc_name);
            }
            for (int j=depth-1; j > 0; j--) {
                if (k == 1 && j%2 == 0) {
                    ancilla_value = (ancilla_qubit_value[j][k-1] == 1) ? -1 : 1;
                    sprintf(anc_name, "%d,", ancilla_value);
                    strcat(label_list, anc_name);
                } else if (k == (depth - 1) && j%2 == 1) {
                    ancilla_value = (ancilla_qubit_value[j][k+1] == 1) ? -1 : 1;
                    sprintf(anc_name, "%d,", ancilla_value);
                    strcat(label_list, anc_name);
                }
                ancilla_value = (ancilla_qubit_value[j][k] == 1) ? -1 : 1;
                sprintf(anc_name, "%d,", ancilla_value);
                strcat(label_list, anc_name);
            }
            if (k%2 == 1) {
                ancilla_value = (ancilla_qubit_value[0][k] == 1) ? -1 : 1;
                sprintf(anc_name, "%d,", ancilla_value);
                strcat(label_list, anc_name);
            }
        }

        // For printing label list:
        strcat(label_list, "\"[");
        char qubit_name[6];
        int first = 1;

        for (int k = 1; k < depth + 1; k++) {
            for (int j = depth; j > 0; j--) {
                if (data_qubit_x_error[j][k] == 1) {
                    if (first == 1) {
                        first = 0;
                    } else {
                        strcat(label_list, ", ");
                    }
                    sprintf(qubit_name, "'X%d%d'", (k-1), (depth-j));
                    strcat(label_list, qubit_name);
                }
                if (data_qubit_z_error[j][k] == 1) {
                    if (first == 1) {
                        first = 0;
                    } else {
                        strcat(label_list, ", ");
                    }
                    sprintf(qubit_name, "'Z%d%d'", (k-1), (depth-j));
                    strcat(label_list, qubit_name);
                }
            }
        }
        strcat(label_list, "]\"");

        MPI_File_write_all(fh, label_list, strlen(label_list), MPI_CHAR, MPI_STATUS_IGNORE);

	}

    MPI_File_close(&fh);

    if (iproc == 0) {
        clock_t end = clock();
        exec_time += (double) (end - begin) / CLOCKS_PER_SEC;
        printf("Samples collected for depth %d, %d in %f seconds on %d processes\n", depth, total_num_iter, exec_time, nproc);
    }

	MPI_Finalize();

	return 0;
}