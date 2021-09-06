"""Module for implementing an Audi Autonomous Driving Dataset (A2D2) DataLoader
designed for performing data fusion betweeen 2D RGB images and 3D Lidar point clouds."""
import numpy as np
from torch.utils.data import Dataset

class A2D2DataLoader(Dataset):
    def __init__(self, dataset, rotation=None, \
                 normalize_xyz=True, normalize_rgb=True, \
                 take_subset=False, convert_to_tensor=True, \
                 target_ids=[]):
        """DataLoader designed for performing data fusion betweeen
        2D RGB images and 3D Lidar point clouds.

        Parameters:
            dataset (dict): Dictionary corresponding to the dataset
                for which fusion will occur. See A2D2 for more details.
            rotation (int): The maximum angle of rotation (in degrees)
                permitted.
            normalize_xyz (bool): Whether to normalize the xyz point cloud
                coordinates to the unit sphere. Defaults to True.
            normalize_rgb (bool): Whether to normalize the RGB pixels values
                to the [0, 1], fp32 range. Defaults to True.
            take_subset (bool): Whether to consider a subset of the semantic
                classes for the dataset. Defaults to False. If set to True,
                select the target ID classes (numeric integers) by setting
                the target_ids parameter.
            convert_to_tensor (bool): Whether to convert the resulting np.ndarray
                objects to PyTorch tensors. Defaults to True.
            target_ids (list): A list of target IDs to consider for constructing this
                dataset, e.g. if only considering a subset of samples. In order to
                take a subset of the samples, take_subset must be set to True.
                Defaults to an empty list, i.e. [].
        """
        # Get IDS
        self.ids = list(dataset.keys())

        # Get rotation and length of dataset
        self.rotation = rotation
        self.N = len(self.ids)

        # Get geometric point cloud data and normalize
        self.xyz = [dataset[ID]['points'] for ID in self.ids]
        self.xyz_norm = self.normalize_xyz()

        # Get rgb data and normalize
        self.rgb = [dataset[ID]['rgb'] for ID in self.ids]
        self.rgb_norm = self.normalize_rgb()

        # Combine xyz and rgb
        self.xyz_rgb = np.hstack((self.xyz, self.rgb))
        self.xyz_rgb_norm = [np.hstack((self.xyz_norm[i], self.rgb_norm[i])) for
                             i in range(self.N)]

        # Get labels
        self.labels = [dataset[ID]['labels'] for ID in self.ids]

        # Get number of points to use
        self.num_points = np.min([len(self.xyz[i]) for i in range(self.N)])
        print("SMALLEST PC POINTS: {}".format(self.num_points))

        if take_subset:
            self.target_ids = target_ids
            # Now get subset
            self.general_dataset, self.target_dataset = \
                self.split_ds_by_classes()
        if convert_to_tensor:
            self.xyz_norm_tensor, self.rgb_norm_tensor, \
            self.xyz_rgb_norm_tensor, self.labels_tensor = \
                self.convert_to_tensor()

    def __getitem__(self, index):
        if self.rotation is not None:
            index_xyz = self.xyz[index]
            angle = np.random.randint(self.rotation[0],
                                      self.rotation[1]) * np.pi / 180
            pointcloud = self.rotate_point_cloud_by_angle(index_xyz, angle)

            return pointcloud, self.labels[index]
        else:
            return self.xyz_rgb_norm_tensor[index], self.labels_tensor[
                index], len(self.xyz_rgb_norm_tensor[index])
