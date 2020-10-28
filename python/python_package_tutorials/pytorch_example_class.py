from PIL import Image
from torch.utils.data import Dataset
import torch
import os


class ImageDataset(Dataset):
    """Image Dataset from files.  Assumes the root_dir contains .JPEG images and a labels file called "val_map.txt".  Allows for an optional transform as well."""

    def __init__(self, root_dir, transform=None):
	all_files = os.listdir[rootdir]       
	image_files = [os.path.join(root_dir, f) if f.endswith(".JPEG") for f in all_files]
	labels_file = "val_map.txt"
	self.transform = transform
	
	# Load all images
	if transform is not None:
		images = [self.transform(Image.open(f)) for f in image_files]
	else:
		images = [Image.open(f) for f in image_files]

	# Image --> label dict
	self.labels = {}
	files2ids = {f: i for i, f in enumerate(image_files)}

	# Read labels
	with open(os.path.join(root_dir, labels_file), "r") as labels:
		line = labels.readline()
		img_name, label = line.split(" ")  # Split by whitespace
		self.labels[files2ids[os.path.join(root_dir,img_name)]] = int(label)
	self.images = {files2ids[f]: I for (f, I) in zip(image_files, images)}

    def __len__(self):
	"""Gives us the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
	"""Method called when we iterate over the dataset, for instance in a training or evalaution loop.  idx is an index for calling a specific image and its associated label."""
        if torch.is_tensor(idx):
            idx = list(idx)
	if isinstance(idx, int):
		return self.images[idx], self.labels[idx]
	else:
		return torch.tensor([self.images[i] for i in idx]), torch.tensor([self.labels[i] for i in idx])

