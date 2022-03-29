/*	2d Lt Brett Martin
 *	Advisor: Dr. Laurence Merkle
 *	Surface code simulation - random-sampled version with replacement
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#define TRI_SIZE 50

struct s_prob {
    int n_x_err, n_z_err;
    float probability, cumulative;
    unsigned long count;    // TODO: Change to double to have higher range
};

int q_comp(const void *p1, const void *p2) {
    float a = ((struct s_prob *)p1)->probability;
    float b = ((struct s_prob *)p2)->probability;
    if (a < b) { return -1; }
    else if (a > b) { return 1; }
    return 0;
}


// Helper function to the Fisher Yates Shuffle function, swaps values at two indices of an array
void swap( int *first, int *second) {
    int buf = *first;
    *first = *second;
    *second = buf;
}


// Fisher Yates Shuffle for randomizing an array of indices
void fisher_yates( int array[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap(&array[i], &array[j]);
    }
}


int main (int argc, char** argv) {

    // Execution time variables:
    double exec_time = 0.0;
    clock_t begin = clock();

	//Inputs:
	const int depth = 7;		// Or code distance
    const int num_samples = 500000;
	const float p_x_error = 0.05;
	const float p_z_error = 0.05;

    // Set this to 1 if you want the sample set to include instances where no errors exist
    int include_no_err = 0;

	//Output:
	//list of records in which each possible combination of data qubit errors is associated with the resulting ancilla qubit values (may be probabilistic), along with the probability of the combination of data qubit errors (and measurement errors?)
	FILE *f = fopen("D:/Documents/AFIT/Research/REPO/QIS/brett.martin/Neural Network/TEST/v2samples-d7-500000.csv", "w+");

    // Generate time-dependent seed for random number generation
    srand(time(NULL));

    const int num_data_qubits = depth * depth;
    int data_qubit_x_error[ depth + 2][ depth + 2 ];
    int data_qubit_z_error[ depth + 2  ][ depth + 2 ];
    int ancilla_qubit_value[ depth + 1 ][ depth + 1 ];

    // Generate pascal's triangle for use in calculating binomial coefficients: O(n^2)
    // When referencing (n k) or "n choose k", just use: pascal_triangle[n][k]
    const int tri_size = 50;
    unsigned long pascal_triangle[ TRI_SIZE ][ TRI_SIZE ];

    for (int row = 0; row < tri_size; row++) {  // This generates a ton of empty space, but it should be okay with such negligible size
        for (int col = 0; col <= row; col++)
        {
            if (row == col || col == 0) {
                pascal_triangle[row][col] = (unsigned long) 1;
            } else {
                pascal_triangle[row][col] = (unsigned long) pascal_triangle[row-1][col] + pascal_triangle[row-1][col-1];
            }
        }
    }

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
    fprintf(f, "Labels\n");
    for (int i=0; i<(depth-1)*(depth+1); i++) { fprintf(f, "1,"); }
    fprintf(f, "\"[]\"\n");

    // Determine the number of unique probabilities, the number of samples generated per probability
    struct s_prob prob_values[(num_data_qubits + 1) * (num_data_qubits + 1)];
    memset(prob_values, 0, sizeof(prob_values));

    int prob_count = 0;
    for (int i = 0; i < num_data_qubits + 1; i++) {
        for (int j = 0; j < num_data_qubits + 1; j++) {
            prob_values[prob_count].n_x_err = i;
            prob_values[prob_count].n_z_err = j;
            prob_values[prob_count].probability = (float) pow((1.0 - p_x_error), num_data_qubits - i) * pow((1.0 - p_z_error), num_data_qubits - j) * pow(p_x_error, i) * pow(p_z_error, j);
            prob_values[prob_count].count = (unsigned long) pascal_triangle[num_data_qubits][i] * pascal_triangle[num_data_qubits][j];
            prob_count++;
        }
    }

    // Sort by ascending probability
    qsort((void*)prob_values, (num_data_qubits + 1) * (num_data_qubits + 1), sizeof(prob_values[0]), q_comp);

    // Add cumulative probabilities from least to most probable
    prob_count = 0;
    float cum_prob = 0.0;
    for (int i = 0; i < num_data_qubits + 1; i++) {
        for (int j = 0; j < num_data_qubits + 1; j++) {
            cum_prob += (float)(prob_values[prob_count].probability * prob_values[prob_count].count);
            prob_values[prob_count].cumulative = cum_prob;
            prob_count++;
        } 
    }

    // If all samples must contain at least one error, lower upper bound of random function to exclude that probability range
    float upper_prob_bound = include_no_err ? 1.0 : (prob_values[(num_data_qubits + 1) * (num_data_qubits + 1) - 2].cumulative - 0.01);

    // Begin main sim outer loop
	for ( int i = 0; i < num_samples; i++ ) {

		// initialize data_qubit_x_error and data_qubit_z_error to be full of FALSE (or 0) values
		memset(data_qubit_x_error, 0, sizeof(data_qubit_x_error));
		memset(data_qubit_z_error, 0, sizeof(data_qubit_z_error));
		memset(ancilla_qubit_value, 0, sizeof(ancilla_qubit_value));

        // Generate random number between 0 and 1 (or upper bound)
        float rand_num = ((float) rand() / RAND_MAX) * upper_prob_bound;
        int sample_idx = 0;
        for (int j = 0; j < (num_data_qubits + 1) * (num_data_qubits + 1); j++) {
            // Index of that element is going to point to the entry in the struct array that will tell me how many of each error will occur, but only if all samples have not been generated
            if (prob_values[j].cumulative >= rand_num) {     // Check if the cumulative value is higher AND if there are enough samples to continue
                sample_idx = j;
                break;
            }
        }

        // Initialize array of length (num_data_qubits) where each element in the array is equal to its index
        int rand_array[num_data_qubits];
        for (int j = 0; j < num_data_qubits; j++) { rand_array[j] = j; }

        // Perform n_x_err and n_z_err iterations of the Fisher Yates Shuffle on the array (where the selected value is at index 0)
        // This will tell me the data qubits that have x errors
        int qubit_idx;
        for (int j = 0; j < prob_values[sample_idx].n_x_err; j++) {
            while (1) {    // Ensure the same error doesn't get written twice
                fisher_yates(rand_array, num_data_qubits);
                qubit_idx = rand_array[0];
                if (data_qubit_x_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] != 1) { break; }
            }
            data_qubit_x_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] = 1;
        }

        for (int j = 0; j < prob_values[sample_idx].n_z_err; j++) {
            while (1) {
                fisher_yates(rand_array, num_data_qubits);
                qubit_idx = rand_array[0];
                if (data_qubit_z_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] != 1) { break; }
            }
            data_qubit_z_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] = 1;
        }

		for ( int j = 0; j < pow( (depth + 1 ), 2) - 1; j++ ) {
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
                    if (first) { first = 0; }
                    else { strcat(label_list, ", "); }
                    sprintf(qubit_name, "'X%d%d'", (k-1), (depth-j));
                    strcat(label_list, qubit_name);
                }
                if (data_qubit_z_error[j][k] == 1) {
                    if (first) { first = 0; }
                    else { strcat(label_list, ", "); }
                    sprintf(qubit_name, "'Z%d%d'", (k-1), (depth-j));
                    strcat(label_list, qubit_name);
                }
            }
        }
        strcat(label_list, "]\"");
        fprintf(f, "%s\n", label_list);

	}

    fclose(f);

    clock_t end = clock();
    exec_time += (double) (end - begin) / CLOCKS_PER_SEC;

    printf("Samples collected for %d, %d in %f seconds", depth, num_samples, exec_time);

	return 0;
}