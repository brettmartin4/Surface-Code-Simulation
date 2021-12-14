/*	2d Lt Brett Martin
 *	Parallel surface code simulation
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

int main (int argc, char** argv) {

	//Inputs:
	int depth = 3;		// Or code distance
	float p_x_error = 0.05;
	float p_z_error = 0.05;

	//Output:
	//list of records in which each possible combination of data qubit errors is associated with the resulting ancilla qubit values (may be probabilistic), along with the probability of the combination of data qubit errors (and measurement errors?)
	FILE *f = fopen("output.csv", "w+");

    // Create CSV headers
    int ancilla = 0;
    for (int k=1; k < depth; k++) {
        if (k%2 == 0) {
            fprintf(f, "Z%d,", ancilla);
		    ancilla += 1;
        }
        for (int j=depth-1; j > 0; j--) {
            if (k == 1 && j%2 == 0) {
                fprintf(f, "X%d,", ancilla);
		        ancilla += 1;
            } else if (k == (depth - 1) && j%2 == 1) {
                fprintf(f, "X%d,", ancilla);
		        ancilla += 1;
            }
            if ((j+k)%2 == 0) {
                fprintf(f, "X%d,", ancilla);
		        ancilla += 1;
            } else {
                fprintf(f, "Z%d,", ancilla);
		        ancilla += 1;
            }
        }
        if (k%2 == 1) {
            fprintf(f, "Z%d,", ancilla);
		    ancilla += 1;
        }
    }
    fprintf(f, "Labels,");
    fprintf(f, "Probability\n");

	int num_data_qubits = depth * depth;
	int data_qubit_x_error[ depth + 2][ depth + 2 ];
	int data_qubit_z_error[ depth + 2  ][ depth + 2 ];
	int ancilla_qubit_value[ depth + 1 ][ depth + 1 ];

	// Must include parallel block decomposition variables when converted to run on HPC
    // Surface code of depth 7 exceeds this value. Find an alternate means to count the total number of loop iterations
	uint64_t total_num_iter = (uint64_t) pow(4, num_data_qubits);			// Total number of outer loop iterations    

	for ( int i = 0; i < total_num_iter; i++ ) {

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

        // Output the ancilla qubit values in proper format
        int ancilla_value;
        for (int k=1; k < depth; k++) {
            if (k%2 == 0) {
                ancilla_value = (ancilla_qubit_value[depth][k] == 1) ? -1 : 1;
                fprintf(f, "%d,", ancilla_value);
            }
            for (int j=depth-1; j > 0; j--) {
                if (k == 1 && j%2 == 0) {
                    ancilla_value = (ancilla_qubit_value[j][k-1] == 1) ? -1 : 1;
                    fprintf(f, "%d,", ancilla_value);
                } else if (k == (depth - 1) && j%2 == 1) {
                    ancilla_value = (ancilla_qubit_value[j][k+1] == 1) ? -1 : 1;
                    fprintf(f, "%d,", ancilla_value);
                }
                ancilla_value = (ancilla_qubit_value[j][k] == 1) ? -1 : 1;
                fprintf(f, "%d,", ancilla_value);
            }
            if (k%2 == 1) {
                ancilla_value = (ancilla_qubit_value[0][k] == 1) ? -1 : 1;
                fprintf(f, "%d,", ancilla_value);
            }
        }

        // For printing label list:
        int max_char = 14 * (depth * depth) + 3;
        char label_list[max_char];
        strcpy(label_list, "\"[");
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
                    sprintf(qubit_name, "'X%d%d'", (k-1), (3-j));
                    strcat(label_list, qubit_name);
                }
                if (data_qubit_z_error[j][k] == 1) {
                    if (first == 1) {
                        first = 0;
                    } else {
                        strcat(label_list, ", ");
                    }
                    sprintf(qubit_name, "'Z%d%d'", (k-1), (3-j));
                    strcat(label_list, qubit_name);
                }
            }
        }
        strcat(label_list, "]\"");
        fprintf(f, "%s\n", label_list);

	}

    fclose(f);

	return 0;
}