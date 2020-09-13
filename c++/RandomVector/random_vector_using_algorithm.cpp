/* Functionality for implementing a random vector using the algorithm header library.
*/

#include <algorithm>
#include "random_vector.h"
#include "float.h"

// Constructor
RandomVector::RandomVector(int size, double max_val) {

    // Initialize vector
    std::vector<double> random_vector(size);

    // Create a normalizing factor (largest number sampled)
    double normalizing_factor = DBL_MIN;

    //  Add elements to it
    for (int i=1; i <= size; ++i) {  // O(N)
        random_vector[i] = rand();  // sample random values
        if (random_vector[i] > normalizing_factor) {  // if new max
            normalizing_factor = random_vector[i];  // update
        }
    }

    // Now normalize and scale by max_val
    for (int i=1; i <= size; ++i) { // O(N)
        random_vector[i] *= max_val / normalizing_factor;
    }

    // Set class field equal to random vector
    this->vect = random_vector;  // get random vector
}

// Prints all elements of the random vector
void RandomVector::print(){
    // Iteratively add all elements of random array to constant output
    for(int i=0; i < this->vect.size(); ++i) {
        std::cout << this->vect[i] << ' ';
    }
    // When finished, end the line
    std::cout << " " << std::endl;

}

// Computes mean of random vector
double RandomVector::mean(){
   // Placeholder value for total
    double total = 0;

   //  Compute total sum of random array
   for (int i=0; i < this->vect.size(); ++i) {
       total += this->vect[i];
   }

   return total / this->vect.size();
}

// Computes maximum of random vector
double RandomVector::max(){
    // Placeholder value for max
    double max_value = *std::max_element(this->vect.begin(), this->vect.end());
    return max_value;
}

// Computes minimum of random vector
double RandomVector::min(){
    // Placeholder value for min
    double min_value = *std::min_element(this->vect.begin(), this->vect.end());
    return min_value;
}

// Computes and prints the histogram of the random vector
void RandomVector::printHistogram(int bins){
    // Determine how much to increment between axes
    double min_val = this->min(); // get min value
    double max_val = this->max();  // get max value
    double interval_delta = (max_val - min_val) / bins;

    // Create array of intervals
    std::vector<double> interval_array(bins);

    // Initialize array of counts
    std::vector<int> bin_counts(bins);

    // Now add values to array iteratively
    for (int i=0; i <= bins; ++i) {
        interval_array[i] = min_val + (i * interval_delta);
    }

    // Iterate over bins
    for (int i=0; i <= (bins - 1); ++i) {

        // Iterate over points in dataset
        for (int j=0; j < this->vect.size(); ++j) {
            // Increment if value is between intervals
            if ((this->vect[j] >= interval_array[i]) && (this->vect[j] <= interval_array[i+1])) {
                ++bin_counts[i];  // Increment count by 1
            }

        }
    }

    // Print the histogram - first get max bin count
    int max_bin_count = DBL_MIN;
    for (int i=0; i < bins; ++i) {
        if (bin_counts[i] > max_bin_count) {
            max_bin_count = bin_counts[i];
        }
    }

    // Iterate over max bin counts, beginning with highest value
    for (int i=max_bin_count; i >= 0; --i) {

        // Iterate over bins
        for (int j=0; j < bins; ++j) {

            // Indicates that bin count[j] >= i
            if (bin_counts[j] >= i) {
                std::cout << std::string(3, '*') << std::string(1, ' ');
            }

            // Indicates that bin count[j] < i
            else {
                std::cout << std::string(3, ' ') << std::string(1, ' ');
            }

            // If last bin, end line
            if (j == (bins - 1)) {
                std::cout << std::endl;
            }
        }
    }
}