import pandas as pd
import numpy as np

# Import custom plotting
import plot_db

# Function to read in our data
def load_data(data_path):
    """Function for loading and normalizing our data."""
    df = pd.read_csv(data_path, names=['feature1', 'feature2',
                                       'label'],  header=None)

    # Get x1 and x2
    feature_1 = df['feature1'].to_list()
    feature_2 = df['feature2'].to_list()

    # Normalize features using function below
    norm_feature_1 = normalize_feature(feature_1)
    norm_feature_2 = normalize_feature(feature_2)

    # Create concatenated features and get labels
    features = np.array([norm_feature_1, norm_feature_2])
    labels = df['label'].to_list()

    # Add features to data frame
    df['norm_x1'] = norm_feature_1
    df['norm_x2'] = norm_feature_2

    return features, labels, df

def normalize_feature(feature):
    """Function for normalizing our features."""
    # Compute mean and standard deviation, and return (x-mu)/std
    mean = np.mean(feature)
    std = np.std(feature)
    return np.divide(np.subtract(feature, mean), std)


def gradient_descent(features, labels, alpha, num_iters):
    """Function for running gradient descent with linear regression on our data."""
    # Initial settings of weights
    weights = [0, 0, 0]

    # Length of dataset
    N = len(features[0])

    # Take 100 gradient steps
    gradient_losses = [0, 0, 0]

    # Take num_iters steps of gradient descent
    for step in range(num_iters):

        # For reach data point, compute the gradients w.r.t. weights and offset
        for x1, x2, y in zip(features[0], features[1], labels):

            # Create "expanded feature dimension for x to account for offset
            x = [1, x1, x2]

            # Make prediction
            pred = weights[0]*x[0] + weights[1]*x[1] + weights[2]*x[2]

            # Compute gradient of loss for linear regression
            for j in range(len(gradient_losses)):
                gradient_losses[j] += (pred-y) * x[j]

        # Update weights using gradients above
        for j in range(len(gradient_losses)):
            weights[j] -= (alpha/N) * gradient_losses[j]

        # Reset gradients of loss after each step
        gradient_losses = [0, 0, 0]

    # Return the weights
    return [weights[0], weights[1], weights[2]]


def main():
    """Function for running the gradient descent algorithm."""
    # Load data and pre-process it
    path = "data2.csv"
    features, labels, df = load_data(path)

    # Learning rates - including our own
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.75]

    # Num iterations
    num_iterations = [100 for i in range(9)] + [1000]

    # Keep track of all final weights for different learning rates
    lines = []

    # Compute weights for each learning rate
    for rate, num_iters in zip(learning_rates, num_iterations):

        # Get weights from gradient descent and add to weights list
        weights = gradient_descent(features, labels, rate, num_iters)
        lines.append([rate, num_iters] + weights)

    # Now write 'lines' to file
    with open('results2.csv', "w") as out_file:
        for line in lines:
            out_file.write("{}, {}, {}, {}, {} \n".format(line[0], line[1],
                                                       line[2], line[3], line[4]))
        out_file.close()

    # Select which weights to use for plotting
    index = -1


    plot_db.visualize_3d(df, lin_reg_weights=lines[index][2:],
                         feat1='norm_x1', feat2='norm_x2', labels='label',
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                         alpha=learning_rates[index], xlabel='age',
                         ylabel='weight', zlabel='height',
                         title='')


if __name__ == "__main__":
    main()