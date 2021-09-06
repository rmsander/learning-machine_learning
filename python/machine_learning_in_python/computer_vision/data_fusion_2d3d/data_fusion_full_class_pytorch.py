"""Module for implementing an Audi Autonomous Driving Dataset (A2D2) DataLoader
 designed for performing data fusion betweeen 2D RGB images and 3D Lidar point clouds.
 This is the FULL class."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
import json
import pickle
import pptk
from pyntcloud import PyntCloud
import copy
import warnings

warnings.filterwarnings('ignore')


# Now let's use this pointnet class
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

    def __len__(self):
        return self.N

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix)

        return rotated_data

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

    def normalize_xyz(self):
        normalized_xyz = []
        for ID in range(len(self.ids)):
            XYZ = np.copy(self.xyz[ID])
            centroid = np.mean(XYZ, axis=0)
            XYZ -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(XYZ) ** 2, axis=-1)))
            XYZ /= furthest_distance
            normalized_xyz.append(XYZ)
        print("XYZ normalized")
        return normalized_xyz

    def normalize_rgb(self):
        normalized_rgb = []
        for ID in range(len(self.ids)):
            RGB = np.copy(self.rgb[ID])
            RGB = np.divide(RGB, 255.0)
            normalized_rgb.append(RGB)
        print("RGB normalized")
        return normalized_rgb

    def convert_to_tensor(self):
        """
        xyz_norm_tensor = torch.tensor(self.xyz_norm)
        rgb_norm_tensor = torch.tensor(self.rgb_norm)
        xyz_rgb_norm_tensor = torch.tensor(self.xyz_rgb_norm)
        labels_tensor = torch.tensor(self.labels)
        """
        xyz_norm_tensor = [torch.tensor(dp) for dp in self.xyz_norm]
        rgb_norm_tensor = [torch.tensor(dp) for dp in self.rgb_norm]
        xyz_rgb_norm_tensor = [torch.tensor(dp) for dp in self.xyz_rgb_norm]
        labels_tensor = [torch.tensor(dp) for dp in self.labels]

        return xyz_norm_tensor, rgb_norm_tensor, xyz_rgb_norm_tensor, \
               labels_tensor

    def split_ds_by_classes(self):
        # Init output data structures
        gen_ds, target_ds = {}, {}

        general_id_indices = [j for j in range(55) if j not in self.target_ids]
        general_id_map = {general_id_indices[k]: k for k in
                          range(len(general_id_indices))}

        # Now make subset and general ID maps and pickle them
        self.target_ids.sort()
        target_id_map = {self.target_ids[i]: i for i in
                         range(len(self.target_ids))}
        target_id_map.update(general_id_map)

        print("Target ID Map: \n {}".format(target_id_map))
        print("General ID Map: \n {}".format(general_id_map))

        # Now pickle these
        f_out = os.path.join(os.getcwd(), "data", "ID_MAPS.pkl")
        with open(f_out, "wb") as f:
            pickle.dump([general_id_indices, general_id_map], f)
            f.close()

        for index in range(self.N):  # Iterate over all images
            FOUND = False
            if index % 10000 == 0:
                print("Iterated through {} files".format(index))
            unique_ids = np.unique(self.labels[index])
            for ID in unique_ids:
                if ID in self.target_ids:
                    labels = self.labels[index]
                    mapped_labels = [target_id_map[labels[j]] for j in
                                     range(len(labels))]
                    target_ds[self.ids[index]] = {'points': self.xyz[index],
                                                  'labels': mapped_labels,
                                                  'rgb': self.rgb[index]}
                    FOUND = True
            if not FOUND:
                gen_ds[self.ids[index]] = {'points': self.xyz[index],
                                           'labels': self.labels[index],
                                           'rgb': self.rgb[index]}
        print(
            "Number of pcs in general: {}, Number of pcs in target: {}".format(
                len(list(gen_ds.keys())), \
                len(list(target_ds.keys()))))
        return gen_ds, target_ds


def create_dataloader_wrapper(f_dataset, normalize_xyz=True,
                              normalize_rgb=True, \
                              take_subset=False, convert_to_tensor=True):
    # Get input dataset
    with open(f_dataset, "rb") as f:
        dataset = pickle.load(f)
        f.close()

    # Instantiate the class object
    dataloader = A2D2DataLoader(dataset, normalize_xyz=normalize_xyz,
                                normalize_rgb=normalize_rgb, \
                                take_subset=take_subset,
                                convert_to_tensor=convert_to_tensor)
    return dataloader


def main():
    # Get input dataset
    f_in = os.path.join(os.getcwd(), "data",
                        "dataset_pc_labels_camera_start_0_stop_10000.pkl")
    dataset = create_dataloader_wrapper(f_in, take_subset=True)
    print("Finished processing dataset")
    # Get general and target datasets
    gen_ds, target_ds = dataset.general_dataset, dataset.target_dataset

    # Now create A2D2 DataLoader Objects based off of these datasets
    gen_dataset = A2D2DataLoader(gen_ds, normalize_xyz=True,
                                 normalize_rgb=True, \
                                 take_subset=False, convert_to_tensor=True)

    # Create output fname
    f_out_general = os.path.join(os.getcwd(), "data",
                                 "PROCESSED_general_dataset_norm_tensor.pkl")

    # Pickle results - general
    with open(f_out_general, "wb") as f:
        pickle.dump(gen_ds, f)
    f.close()

    print("Pickled general processed dataset to {}".format(f_out_general))

    target_dataset = A2D2DataLoader(target_ds, normalize_xyz=True,
                                    normalize_rgb=True, \
                                    take_subset=False, convert_to_tensor=True)

    # Create output fname
    f_out_target = os.path.join(os.getcwd(), "data",
                                "PROCESSED_target_dataset_norm_tensor.pkl")

    # Pickle results - general
    with open(f_out_target, "wb") as f:
        pickle.dump(gen_ds, f)
        f.close()

    print("Pickled target processed dataset to {}".format(f_out_target))


if __name__ == "__main__":
    main()
