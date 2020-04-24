from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2 as cv

def read_and_process_image(fpath):
    """A function to read in our image, and turn it into an N x 3 vector, where
    N is the number of pixels in the image."""

    # Load image from path and get shape
    image = cv.imread(fpath)
    H, W, C = image.shape

    # Reshape image into (N, 3) vector
    image_vector = image.reshape((H * W, C))

    return image_vector, (H, W, C)

def k_means_clustering(image_vec, num_clusters=3):
    """Function for clustering our image vector in RGB vector space using the
    K-Means clustering algorithm."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(image_vec)
    return kmeans.labels_

def clusters_to_image(kmeans_labels, old_shape):
    """A function for reconstructing our image using the K-Means clusters."""

    # Puts image back into (H, W, 3)
    cluster_image = kmeans_labels.reshape(old_shape)

    return cluster_image

def plot_matrix(img, title='', num_clusters=3,
                out_file_path='kmeans_segment_img_{}_clusters.png'):
    """Function for plotting the color matrix corresponding to the integer
    value of each cluster."""

    # Create color map
    cmap = colors.ListedColormap(['k', 'b', 'y', 'g', 'r'])  # Add more here

    # Plot and save image using the color_path
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.imsave(out_file_path.format(num_clusters), img)

def main():
    """Main function for k-means clustering-based image segmentation."""

    # Read in and process image.
    fpath = 'trees.png'
    image_vector, old_shape = read_and_process_image(fpath)

    # Iterate for three, four, and five clusters
    for k in [3, 4, 5]:

        # Cluster data, convert back to original image space, and save figure
        cluster_labels = k_means_clustering(image_vector, num_clusters=k)
        cluster_image = clusters_to_image(cluster_labels, old_shape[0:2])
        plot_matrix(cluster_image,
                    title='K-Means Segmentation for {} Clusters'.format(k),
                    num_clusters=k)

if __name__ == "__main__":
    main()