/*	2d Lt Brett Martin
 *	Advisor: Dr. Laurence Merkle
 *	Surface code simulation - random-sampled version without replacement
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

#define TRI_SIZE 50


// Struct used in calculating and sorting all possible probabilities along with the number of samples per probability and the respective number of x and z errors
struct s_prob {
    int n_x_err, n_z_err, n_total_err;
    float probability, cumulative;
    unsigned long count, collected;
};


// Struct used to keep track of the x and z error syndrome for a given sample in the form of bit strings
struct error_syndrome {
    uint64_t x_syndrome, z_syndrome;
};


// Helper function passed into the quicksort algorithm to sort the struct s_prob by the probability values
int q_comp( const void *p1, const void *p2 ) {
    float a = ( (struct s_prob *) p1 )->probability;
    float b = ( (struct s_prob *) p2 )->probability;
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


// Begin main function
int main ( int argc, char** argv ) {
/*
    char *d = argv[1];
    int depth = atoi(d);
    if (depth != 3 && depth != 5 && depth != 7) {
        printf("Depth must be 3, 5, or 7. Exiting program...\n");
        return 0;
    }

    char *s = argv[2];
    int num_samples = atoi(s);
    if (num_samples < 10 || num_samples > 500000) {
        printf("Number of samples falls outside safe range. Recompile program with different values to continue outside this range. Returning...\n");
        return 0;
    }*/

    // Execution time variables:
    double exec_time = 0.0;
    clock_t begin = clock();

	//Inputs:
	const int depth = 7;		// Or code distance
    const int num_samples = 10000;
	const float p_x_error = 0.05;
	const float p_z_error = 0.05;

    // Set this to 1 if you want the sample set to include instances where no errors exist
    int include_no_err = 0;

	//Output:
	//list of records in which each possible combination of data qubit errors is associated with the resulting ancilla qubit values (may be probabilistic), along with the probability of the combination of data qubit errors (and measurement errors?)
	FILE *f = fopen("D:/Documents/AFIT/Research/REPO/QIS/brett.martin/HPC Simulation/v3samples-d7-1000.csv", "w+");

    // Generate time-dependent seed for random number generation
    srand(time(NULL));

    // Variables used in the generation of sample data
    int num_data_qubits = depth * depth;
	int data_qubit_x_error[ depth + 2][ depth + 2 ];
	int data_qubit_z_error[ depth + 2  ][ depth + 2 ];
	int ancilla_qubit_value[ depth + 1 ][ depth + 1 ];

    // Generate pascal's triangle for use in calculating binomial coefficients ( time O(n^2) )
    // When referencing (n k) or "n choose k", just use: pascal_triangle[n][k]
    const int tri_size = 50;
    unsigned long pascal_triangle[ TRI_SIZE ][ TRI_SIZE ];

    for (int row = 0; row < tri_size; row++) {  // This generates a ton of empty space, but it should be okay with such negligible size
        for (int col = 0; col <= row; col++)
        {
            if (row == col || col == 0) { pascal_triangle[row][col] = (unsigned long) 1; }
            else { pascal_triangle[row][col] = (unsigned long) pascal_triangle[row-1][col] + pascal_triangle[row-1][col-1]; }
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
            prob_values[prob_count].n_total_err = i + j;
            prob_values[prob_count].n_x_err = i;
            prob_values[prob_count].n_z_err = j;
            prob_values[prob_count].probability = (float) pow((1.0 - p_x_error), num_data_qubits - i) * pow((1.0 - p_z_error), num_data_qubits - j) * pow(p_x_error, i) * pow(p_z_error, j);
            prob_values[prob_count].count = (unsigned long) pascal_triangle[num_data_qubits][i] * pascal_triangle[num_data_qubits][j];
            prob_count++;
        }
    }

    // Sort by ascending probability and calculate cumulative probs
    qsort((void*)prob_values, (num_data_qubits + 1) * (num_data_qubits + 1), sizeof(prob_values[0]), q_comp);

    prob_count = 0;
    float cum_prob = 0.0;
    for (int i = 0; i < num_data_qubits + 1; i++) {
        for (int j = 0; j < num_data_qubits + 1; j++) {
            cum_prob += (float)(prob_values[prob_count].probability * prob_values[prob_count].count);
            prob_values[prob_count].cumulative = cum_prob;
            prob_count++;
        }
    }

    // Consumes ~156.25 KB of heap space at max of depth=7, so this should be safe. Consider virtual mem if this becomes an issue when running on HPC
    // Will contain which error combinations exist for each possible probability. Searching it would take O(n) later on
    struct error_syndrome combinations[num_samples];
    memset(combinations, 0, sizeof(combinations));
    int combo_ctr = 0;

    // If all samples must contain at least one error, lower upper bound of random function to exclude that probability range
    float upper_prob_bound = include_no_err ? 1.0 : (prob_values[(num_data_qubits + 1) * (num_data_qubits + 1) - 2].cumulative - 0.01);
    prob_values[(num_data_qubits + 1) * (num_data_qubits + 1) - 1].collected = include_no_err ? 0 : 1;

    int max_idx = (num_data_qubits + 1) * (num_data_qubits + 1) - 2;

    // Begin main sim outer loop
    float rand_num;
    int sample_idx, sample_found, qubit_idx, match_found, ancilla_value;
    int rand_array[num_data_qubits];
    uint64_t x_syndrome_bit_string, z_syndrome_bit_string;

    int max_char = 14 * (depth * depth) + 3;
    char label_list[max_char];
    char qubit_name[6];
    int first;

	for (int i = 0; i < num_samples; i++) {

		// initialize data_qubit_x_error and data_qubit_z_error (and others) to be full of FALSE (or 0) values
		memset(data_qubit_x_error, 0, sizeof(data_qubit_x_error));
		memset(data_qubit_z_error, 0, sizeof(data_qubit_z_error));
		memset(ancilla_qubit_value, 0, sizeof(ancilla_qubit_value));

        // Begin Fisher Yates Shuffle to generate data qubit x and z errors
        sample_found = 0;
        while(!sample_found) {
            // Generate random number between 0 and 1 (or upper bound)
            rand_num = ((float) rand() / RAND_MAX) * upper_prob_bound;
            sample_idx = 0;
            for (int j = 0; j < (num_data_qubits + 1) * (num_data_qubits + 1); j++) {
                // Index of that element is going to point to the entry in the struct array that will tell me how many of each error will occur, but only if all samples have not been generated
                if (prob_values[j].cumulative >= rand_num && prob_values[j].count > prob_values[j].collected) {     // Check if the cumulative value is higher AND if there are enough samples to continue
                    sample_idx = j; 
                    prob_values[j].collected++;
                    sample_found = 1;
                    // Check if collected is equal to the count and whether the cumulative is the upper bound. If so, set the new upper bound for rand_num
                    if (max_idx == j && prob_values[j].count == prob_values[j].collected) {
                        max_idx--;
                        upper_prob_bound = prob_values[max_idx].cumulative - 0.01;
                    }
                    break;
                }
            }
        }

        // Initialize array of length (num_data_qubits) where each element in the array is equal to its index
        for (int j = 0; j < num_data_qubits; j++) { rand_array[j] = j; }

        // Perform n_x_err and n_z_err iterations of the Fisher Yates Shuffle on the array (where the selected value is at index 0)
        // This will tell me the data qubits that have x errors
        while(1) {
            x_syndrome_bit_string = 0;   // This value will contain alternating x and z errors (starting with x, and starting at the LSB with data qubit 0)
            z_syndrome_bit_string = 0;
            memset(data_qubit_x_error, 0, sizeof(data_qubit_x_error));
		    memset(data_qubit_z_error, 0, sizeof(data_qubit_z_error));
            for (int j = 0; j < prob_values[sample_idx].n_x_err; j++) {
                while (1) {    // Ensure the same error doesn't get written twice
                    fisher_yates(rand_array, num_data_qubits);
                    qubit_idx = rand_array[0];
                    if (data_qubit_x_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] != 1) { break; }
                }
                data_qubit_x_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] = 1;
                x_syndrome_bit_string = (uint64_t)(1 << qubit_idx) | x_syndrome_bit_string;
            }

            for (int j = 0; j < prob_values[sample_idx].n_z_err; j++) {
                while (1) {
                    fisher_yates(rand_array, num_data_qubits);
                    qubit_idx = rand_array[0];
                    if (data_qubit_z_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] != 1) { break; }
                }
                data_qubit_z_error[ (qubit_idx/depth) + 1 ][ (qubit_idx%depth) + 1 ] = 1;
                z_syndrome_bit_string = (uint64_t)(1 << qubit_idx) | z_syndrome_bit_string;
            }

            // Check list of error syndromes to find match
            match_found = 0;
            for (int j = 0; j < combo_ctr; j++) {
                if (qubit_idx == (num_data_qubits + 1) * (num_data_qubits + 1) - 1) { break; }
                if (combinations[j].x_syndrome == x_syndrome_bit_string && combinations[j].z_syndrome == z_syndrome_bit_string) {
                    match_found = 1;
                    break;
                }
            }

            if (!match_found) {
                combinations[combo_ctr].x_syndrome = x_syndrome_bit_string;
                combinations[combo_ctr].z_syndrome = z_syndrome_bit_string;
                combo_ctr++;
                break;
            }

        }

        // Calculate ancilla qubit values
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
        strcpy(label_list, "\"[");
        first = 1;

        // Print labels with any data qubit errors present
		for (int k = 1; k < depth + 1; k++) {
            for (int j = depth; j > 0; j--) {
                if (data_qubit_x_error[j][k] == 1) {
                    if (first == 1) { first = 0; }
                    else { strcat(label_list, ", "); }
                    sprintf(qubit_name, "'X%d%d'", (k-1), (depth-j));
                    strcat(label_list, qubit_name);
                }
                if (data_qubit_z_error[j][k] == 1) {
                    if (first == 1) { first = 0; }
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