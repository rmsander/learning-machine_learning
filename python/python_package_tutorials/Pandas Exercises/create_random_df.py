import pandas as pd
import numpy as np
import string


def make_csv_table(n_products, n_features, out_file):
	# Create a DataFrame:

	# Step 1: Create a dictionary with arrays
	data = np.random.randint(low=0, high=10, size=(n_products, n_features))
	letters = list(string.ascii_lowercase)[:data.shape[1]]
	df_dict = {}
	for i in range(data.shape[1]):
		df_dict[letters[i]] = data[:, i]

	# Step 2: Convert to a DataFrame
	df = pd.DataFrame(df_dict)
	
	# Step 3: Save as Excel file
	df.to_csv(out_file)
	print("Saved DataFrame to: {}".format(out_file))
		


def main():
	N_PRODUCTS = 10
	N_FEATURES = 10
	OUTFILE = "data.csv"
	make_csv_table(N_PRODUCTS, N_FEATURES, OUTFILE)


if __name__ == "__main__":
	main()
