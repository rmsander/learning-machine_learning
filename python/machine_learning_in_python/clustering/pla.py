import pandas as pd

# Import custom plotting
import plot_db

def load_data(data_path):
    """Function for loading and pre-processing our input data."""
    df = pd.read_csv(data_path, names=['feature1', 'feature2', 'label'],
                     header=None)
    feature_1 = df['feature1'].to_list()
    feature_2 = df['feature2'].to_list()
    features = [[feature_1[i], feature_2[i]] for i in range(len(feature_1))]
    labels = df['label'].to_list()
    return features, labels, df

def perceptron(features, labels):
    """Function implementing the Perceptron learning algorithm."""
    # First, we want to initialize our weights
    weights = [0, 0]
    offset = 0

    # Begin with this condition false
    converged = False

    # Just for keeping track of
    total_mistakes = 0

    # Output weights initialization
    weights_csv = []

    # Run until convergence (for linearly separable dataset)
    while not converged:
        converged = True

        # Iterate over features
        for x, y in zip(features, labels):
            pred = weights[0] * x[0] + weights[1] * x[1] + offset
            if pred * y <= 0:  # If we make an incorrect prediction
                total_mistakes += 1
                converged = False
                weights[0] += y*x[0]
                weights[1] += y*x[1]
                offset += y
                weights_csv.append([weights[0], weights[1], offset])
        print("WRONG: {}".format(total_mistakes))

    # Write the list of output weights to a csv
    with open("results1.csv", "w") as out_file:
        for line in weights_csv:
            out_file.write("{}, {}, {} \n".format(str(line[0]), \
                                               str(line[1]), str(line[2])))
        out_file.close()

    # We've found a linearly separable model
    return weights, offset

def main():
    """Function for loading data, running the perceptron learning algorithm,
    and saving the found weights to results1.csv."""
    # Load data
    path = "data1.csv"
    features, labels, df = load_data(path)

    # Compute weights
    weights, offset = perceptron(features, labels)
    print("Weights are: {} \n"
          "Offsets are: {}".format(weights, offset))

    # Visualize results
    plot_db.visualize_scatter(df, feat1='feature1', feat2='feature2',
                              labels='label', weights=[weights[0], weights[1], offset],
                              title="Perceptron On Linearly Separable Dataset")

if __name__ == "__main__":
    main()