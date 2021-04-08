def is_nice(some_string): 
	string_counts = {} 
	for c in some_string: # Iterate over characters 
		if c in string_counts: 
			string_counts[c] += 1 # Already added, increment by 1 
		else: 
			string_counts[c] = 1 # Added for the first time 

	# Now check to see if all are equal 
	first_val = 0  # Placeholder, value not important
	for i, (c, count) in enumerate(string_counts.items()): # Loops jointly over keys and values 
		if i == 0: 
			first_val = count 
		if count != first_val: # If any of the elements don't equal the first value 
			return False 
	# If we make it this far, all counts are equal, since all counts equal the first count 
	return True
