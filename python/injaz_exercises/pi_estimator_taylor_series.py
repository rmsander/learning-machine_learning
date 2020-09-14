import numpy as np

def pi_estimator(tolerance, verbose=False):
	"""Function for estimating pi using Taylor series.
	
	Parameters: 
		tolerance (float): A floating-point decimal corresponding to the percent tolerance of error for our approximation.
		verbose (bool):  Whether to display line-by-line error and estimate values.

	Returns:	
		pi_est (float):  Estimate of pi after convergence is achieved.
		err (float):  Percent error after estimate converges. 
	"""
	# Initialize variables
	pi_est = 4  # Initial value of pi
	n = 1  # Initial value of n
	converged = False

	# Increment estimator until we converge
	while not converged:
		err = 100 * np.abs(np.pi - pi_est) / np.pi  # Compute percent error
		if verbose:
			print("Error is: {}".format(err))
		if err > tolerance:  # Add another term if convergence has not yet occurred
			pi_est += (4 * ((-1) ** ((n+1) + 1))) / ((2 * (n+1)) - 1)
			if verbose:
				print("Error too great - adding more terms.  New value of pi is: {}".format(round(pi_est, 10)))
		elif err < tolerance:  # Otherwise, convergence has occurred
			converged = True
		
		# Increment term counter		
		n += 1

	return pi_est, err

def main():
	pi_est, err = pi_estimator(0.0001)
	print("Estimate of pi: {}".format(pi_est))
	print("Percent error: {}".format(err))

if __name__ == "__main__":
	main()
