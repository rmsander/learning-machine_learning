// Monte Carlo sampling from synthetic data.  We'll take samples of sample statistics (mean and variance), and analyze their distributions over time.

foreach sample_size in 8 32 128 {
  display `sample_size'

  // Step 3: Generate the random distribution
  matrix sample_means = J(500, 1, 0) // Means of our sampled distributions
  matrix sample_variances = J(500, 1, 0) // Variance of our sampled distributions

  // Iterate from 1 - 500
  forval i = 1(1)500 {

      // Step 1: Set number of observations     
      set obs `sample_size'

      // Step 2: Set random seed for RNG (Random Number Generator)
      set seed 9803`i'

      // Step 3: Generate the samples
      generate sample = rnormal(0, 1)  // This creates a variable called sample with sample size equal to sample_size variable above, with mean 0 and variance 1
      summarize sample

      // Add mean and variance of summary statistics to array
      matrix sample_means[`i', 1] = r(mean)  // Set i^th element of column vector of means
      matrix sample_variances[`i', 1] = r(sd)^2  // Set i^th element of column vector of variances

      // Drop variables to avoid replacement errors
      drop sample
  }
}

