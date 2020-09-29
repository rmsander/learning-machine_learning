import pandas as pd
import os
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--excel_file", type=str, help="Name of the file (without prepended path) that we are reading")
	parser.add_argument("-n", "--pset_number", type=str, help="Problem Set number (e.g. 1, 2, ..., 5)")
	return parser.parse_args()


def main():

	# Parse CLI args and get path to sheet
	args = parse_args()
	excel_path = os.path.join(os.getcwd(), args.excel_file)

	# Load DataFrames for grades and feedback, respectively
	df_grades = pd.read_excel(excel_path, sheet_name="Grades_{}".format(args.pset_number), skiprows=1)
	df_feedback = pd.read_excel(excel_path, sheet_name="Comments_{}".format(args.pset_number))

	# Create feedback
	os.mkdir(os.path.join(os.getcwd(), "grade_breakdown_ps{}".format(args.pset_number)))
	os.mkdir(os.path.join(os.getcwd(), "student_feedback_ps{}".format(args.pset_number)))

	# Create placeholder files for each student
	dest_path_grades = os.path.join(os.getcwd(), "grade_breakdown_ps{}".format(args.pset_number), "{}_grade_breakdown.txt")
	dest_path_feedback = os.path.join(os.getcwd(), "student_feedback_ps{}".format(args.pset_number), "{}_feedback.txt")

	# Get keys
	keys_grades = df_grades.keys()[1:]  # First one is student names and is not needed
	keys_feedback = df_feedback.keys()[1:] # First one is student names and is not needed

	# Iterate through data for filling in grades
	for index, row in df_grades.iterrows():
		with open(dest_path_grades.format(row[0]), "w") as ff:
			for field, item in zip(keys_grades, row[1:]):
				ff.writelines([str(field), "\n"])
				if str(item) != "nan":
					ff.writelines([str(item), "\n"])
				else:
					ff.writelines(["Good job!", "\n"])
				ff.writelines(["\n\n"])
			ff.close()

	# Iterate through data for filling in feedback
	for index, row in df_feedback.iterrows():
		with open(dest_path_feedback.format(row[0]), "w") as ff:
			for field, item in zip(keys_feedback, row[1:]):
				ff.writelines([str(field), "\n"])
				if str(item) != "nan":
					ff.writelines([str(item), "\n"])
				else:
					ff.writelines(["Good job!", "\n"])
				ff.writelines(["\n\n"])
			ff.close()

	print("Done creating grade and feedback files.  \n Grade breakdown files can be found at: \n {} \n AND feedback files can be found at: \n {}".format(os.path.join(os.getcwd(), "grade_breakdown_ps{}".format(args.pset_number)), os.path.join(os.getcwd(), "student_feedback_ps{}".format(args.pset_number))))

if __name__ == "__main__":
	main()
