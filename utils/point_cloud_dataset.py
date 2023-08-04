import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from pathlib import Path
from laspy import read
import numpy as np
from itertools import compress
import open3d as o3d

from utils.ocnn_custom_utils import CustomTransform


def normalize_i(intensity_vals):
    i_norm = ((intensity_vals - min(intensity_vals)) / (max(intensity_vals) - min(
        intensity_vals))) * 20  # Multiply by 20 so that intensity vals take on similar range to biomass vals
    return i_norm


def read_las_to_np(las_fpath, normalize_intensity=True, use_ground_points=None, centralize_coords=True):
    """
    IMPORTANT: format of np array storing lidar data returned has the following format:
    N x C
    Where N is the numer of points and C is the number of columns
    The current columns included in the np array by index are:
    0 - X
    1 - Y
    2 - Z
    3 - Intensity (Normalized or Raw, depending on argument)
    4 - Return Number
    5 - Classification
    6 - Scan Angle Rank
    7 - Number of Returns

    :param centralize_coords: whether to make all coords relative to center of point cloud (center is 0,0,0)
    :param las_fpath: filepath to las file
    :param normalize_intensity: whether to normalize intensity values
    :param use_ground_points: height below which to remove points, specify as None to use no height filter
    :return: point cloud numpy array
    """

    # Read LAS for given plot ID
    inFile = read(las_fpath)

    # Correct for difference in file naming for scan angle (some LAS files call it scan_angle)
    try:
        scan_angle = inFile.scan_angle_rank
    except AttributeError:
        try:
            scan_angle = inFile.scan_angle
        except AttributeError:
            raise Exception("Issue with scan angle name in LAS file...")

    # Get coords coordinates
    points = np.vstack([inFile.x,
                        inFile.y,
                        inFile.z,
                        inFile.intensity,
                        inFile.return_number,
                        inFile.classification,
                        scan_angle,
                        inFile.number_of_returns
                        ]).transpose()

    if use_ground_points:
        pass
    else:
        # Filter the array by dropping all rows with a value of 2 (ground point) in the Classification column
        points = points[points[:, 5] != 2]

    # Normalize Intensity
    if normalize_intensity:
        points[:, 3] = normalize_i(points[:, 3])

    if centralize_coords == True:
        # Centralize coordinates
        points[:, 0:3] = points[:, 0:3] - np.mean(points[:, 0:3], axis=0)

    # Check for NANs and report
    if np.isnan(points).any():
        raise ValueError('NaN values in input point cloud!')
    return points


def duplicate_until_n_poi(points, n_poi_threshold, mean=0, stddev=0.01):
    """
    Duplicates points in the input point cloud `points` until the number of points in the array reaches the specified
    `n_poi_threshold`. The function randomly selects a point (row) from the original point cloud and adds a new point
    with the same coordinates plus some Gaussian noise. The amount of noise is controlled by the `mean` and `stddev`
    parameters. The duplicated point is then added to the original point cloud using `np.vstack`. This process is repeated
    until the target number of points is reached.

    :param points: input point cloud with shape N x C
    :param n_poi_threshold: target number of points
    :param mean: mean value along X,Y,Z axes (should be almost exactly zero since coordinates are normalized when las files are loaded)
    :param stddev: standard deviation from mean from which to add noise
    :return:
    """

    # Continue to duplicate points until correct number of points is reached
    while points.shape[0] < n_poi_threshold:
        # randomly select point (row) to duplicate and add to array
        duplicate_idx = random.sample(range(0, points.shape[0]), 1)
        duplicated_point = points[duplicate_idx, :]

        # Add small amount of noise to XYZ (keep other attributes the same)
        noise = np.random.normal(mean, stddev, size=(1, 3))
        duplicated_point[:, 0:3] = duplicated_point[:, 0:3] + noise

        # Add duplicated point to original point cloud
        points = np.vstack([points, duplicated_point])

    return points


def fps(points, n_poi_threshold):

    """
    Applies farthest point sampling algorithm to downsample point cloud.
    Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html
    :param points: input points with shape N x C
    :param n_poi_threshold: target num of points
    :return: fps downsampled points with shape N x C
    """

    # Get points index values
    idx = np.arange(len(points))

    # Initialize use_idx
    use_idx = np.zeros(n_poi_threshold, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of n_poi_threshold samples
    for i in range(1, n_poi_threshold):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with the largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)

    # Select points to keep and return downsampled point cloud
    fps_points = points[use_idx, :]

    return fps_points


def random_downsample_or_duplicate(points, n_poi_threshold):
    """
    Homogenizes point clouds by selecting random points to remove

    :param points: input point cloud with shape N x C
    :param n_poi_threshold: Threshold for the number of points in the input file
    :return: randomly downsampled or up-duplicated point cloud (referred to as homogenized points)
    """

    # Resample number of points to n_poi_threshold if larger than threshold
    if points.shape[0] > n_poi_threshold:
        use_idx = np.random.choice(points.shape[0], n_poi_threshold, replace=False)

        homogenized_points = points[use_idx, :]

    # If num of points is lower than the threshold, duplicate points with a bit of noise until correct num is reached
    elif points.shape[0] < n_poi_threshold:
        homogenized_points = duplicate_until_n_poi(points, n_poi_threshold)

    else:
        homogenized_points = points

    return homogenized_points


def fps_or_duplicate(points, n_poi_threshold):
    """
    If npoi > n_poi_threshold, applies farthest point sampling algorithm to downsample point cloud.
    If  npoi < n_poi_threshold, duiplicates points with a small amount of noise to reach n_poi_threshold.
    Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html
    :param points: input points with shape N x C
    :param n_poi_threshold: target num of points
    :return: fps downsampled/duplicated points with shape N x C
    """

    if points.shape[0] > n_poi_threshold:
        homogenized_points = fps(points, n_poi_threshold)

    elif points.shape[0] < n_poi_threshold:
        homogenized_points = duplicate_until_n_poi(points, n_poi_threshold)

    else:
        homogenized_points = points

    return homogenized_points


# Data augmentation based off: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

def rotate_points(coords):
    rotation = np.random.uniform(-180, 180)
    # Convert rotation values to radians
    rotation = np.radians(rotation)

    # Rotate point cloud
    rot_mat = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1],
        ]
    )

    aug_coords = coords
    aug_coords[:, :3] = np.matmul(aug_coords[:, :3], rot_mat)
    return aug_coords


def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Modified from original fn to also jitter the intensity, and leave the Return Number, Classification,
        Scan Angle Rank, and Number of Returns as original values
        Input:
          N x C array, original point cloud
        Return:
          N x C array, jittered point cloud
    """
    N, C = points.shape
    assert (clip > 0)
    jittered_points = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_points += points

    # Replace og xyz with jittered xyz (leave other attributes alone)
    points[:, 0:3] = jittered_points[:, 0:3]

    return points


def shift_point_cloud(points, shift_range=0.1):
    """ Randomly shift point cloud.
        Input:
          N x C array, original point cloud
        Return:
          N x C array, shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    points[:, 0:3] = points[:, 0:3] + shifts

    return points


def shuffle_points(points):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Input:
            N x C array
        Output:
            N x C array
    """
    idx = np.arange(points.shape[0])
    np.random.shuffle(idx)
    return points[idx, :]


def augment_point_cloud(points, do_shuffle_points=True):
    if do_shuffle_points:
        points = shuffle_points(points)
    points = jitter_point_cloud(points)
    points = rotate_points(points)
    points = shift_point_cloud(points)
    return points


def estimate_point_normals(points, visualize=False):
    """
    Uses open3d to estimate point normals.
    The "normals" denote the x,y,z components of the surface normal vector at each point
    :param dtype: Data type with which to use for tensor version of point cloud. Default is float64.
    :param points: Input point cloud numpy array with N x C dims.
    :param visualize: Whether to visualize the estimated point normals in a stat.
    :param device: device to hold data. Default is CPU.
    :return: Point cloud with estimated normals.
    """

    # Select coordinates
    xyz = points[:, 0:3]

    # Create an empty point cloud
    pcd = o3d.geometry.PointCloud()

    # Pass xyz coords to point cloud class
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Estimate normals
    pcd.estimate_normals()

    # Convert normals to numpy array
    normals = np.asarray(pcd.normals)

    # Bind normals to original point cloud
    points_w_normals = np.concatenate((points, normals), axis=1)

    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    return points_w_normals


class PointCloudsInFilesPreSampled(Dataset):
    """Point cloud dataset where one sample is a file."""

    def __init__(self, cfg, set='train', partition=None, augment_data=False, files=None):

        """
        IMPORTANT: format of np array storing lidar data returned has the following format:
        N x C
        Where N is the numer of points and C is the number of columns
        The current columns included in the np array by index are:
            0 - X
            1 - Y
            2 - Z
            3 - Intensity (Normalized)
            4 - Return Number
            5 - Classification
            6 - Scan Angle Rank
            7 - Number of Returns

            *If normals are estimated ('use_normals' item in config dict)
            8 - x component of normals
            9 - y component of normals
            10 - z component of normals

        Args:
           cfg: Dictionary which includes config items
           set: train, val, or test split indication
        """

        # Specify key attributes ----------------------------------------------------------------------------------------
        self.set = set
        self.augment_data = augment_data

        # List files ---------------------------------------------------------------------------------------------------
        if self.augment_data == True:
            # Use specified input files
            self.files = files
        else:
            # Get a list of all files in the directory
            self.files = list(Path(cfg['pc_data_path']).glob("*" + cfg['in_filetype']))
            assert len(self.files) > 0

            # Filter files based on target dataset(s) ------------------------------------------------------------------
            # Get dataset source for each file
            dataset_ID = []
            for i in range(0, len(self.files), 1):
                dataset_ID.append(os.path.basename(self.files[0]).split(".")[0][0:2])
            # Convert to pandas series
            dataset_ID = pd.Series(dataset_ID, dtype=str)
            # Determine whether to keep each file based on dataset ID
            dataset_filter = dataset_ID.isin(cfg['use_datasets']).tolist()
            # Filter files to target dataset(s)
            self.files = list(compress(self.files, dataset_filter))

            # Extract the split files belong to (train, test, or val)---------------------------------------------------
            splits = []
            for i in range(0, len(self.files), 1):
                splits.append(os.path.basename(self.files[i]).split(".")[0].split("_")[1])
            # Convert to pandas series
            splits = pd.Series(splits, dtype=str)
            # Determine whether to keep each file based on split string
            splits_filter = splits.isin([self.set]).tolist()
            # Filter files to target dataset(s)
            self.files = list(compress(self.files, splits_filter))

        # Determine whether to partition data --------------------------------------------------------------------------
        if partition is not None:
            self.files = random.sample(self.files, round(len(self.files) * partition))

        # Load biomass data reference data -----------------------------------------------------------------------------
        self.input_table = pd.read_csv(cfg['ref_data_path'], sep=",", header=0)

        # Attach config dict -------------------------------------------------------------------------------------------
        self.cfg = cfg

        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load point cloud ----------------------------------------------------------------------------------------------

        filepath = str(self.files[idx])

        # Read in point cloud from LAS file
        if os.path.basename(filepath).split(".")[1] == "las":
            points = read_las_to_np(filepath,
                                    normalize_intensity=True,
                                    use_ground_points=self.cfg['use_ground_points'],
                                    centralize_coords=True)
        else:
            raise Exception(f'{os.path.basename(filepath).split(".")[1]} file type not supported')

        # If using all points (var is a string), pass, otherwise, downsample to specified number of points (var is int)
        if 'OCNN' not in self.cfg['model_name']:
            # Only use random downsampling on train set since this adds data augmentation
            if self.set == "train":
                # Homogenize point cloud to correct number of points using random down sampling or duplication with noise
                points = random_downsample_or_duplicate(points, n_poi_threshold=self.cfg['num_points'])
            # For val and test sets -> farthest point sampling -> keeps more accurate representation of point cloud
            else:
                points = fps_or_duplicate(points, n_poi_threshold=self.cfg['num_points'])

            points = points[:, 0:3]

        # Apply data augmentation if required
        if self.augment_data:
            # Apply data augmentation
            points = augment_point_cloud(points)

        if self.cfg['use_normals']:
            points = estimate_point_normals(points)
            normals = points[:, 8:11]
        else:
            normals = None

        # Load biomass data --------------------------------------------------------------------------------------------

        # Get plot ID from filename
        PlotID = os.path.basename(self.files[idx]).split(".")[0].split("_")[0]

        # Load the target values (either component biomass absolute values or proportions)
        if self.cfg['target'] == "biomass_comps":
            # Extract bark, branch, foliage, wood values for the correct plot ID
            bark_z = self.input_table.loc[self.input_table["PlotID"] == PlotID]["bark_z"].values[0]
            branch_z = self.input_table.loc[self.input_table["PlotID"] == PlotID]["branch_z"].values[0]
            foliage_z = self.input_table.loc[self.input_table["PlotID"] == PlotID]["foliage_z"].values[0]
            wood_z = self.input_table.loc[self.input_table["PlotID"] == PlotID]["wood_z"].values[0]
            # Combine z targets into a list
            target = [bark_z, branch_z, foliage_z, wood_z]
        else:
            target = self.input_table.loc[self.input_table["PlotID"] == PlotID]["total_Mg_ha"].values[0]

        # Convert points to tensor -------------------------------------------------------------------------------------

        if 'OCNN' in self.cfg['model_name']:

            # Determine which additional features to use from from lidar
            features = points[:, 3:8] if self.cfg['ocnn_use_additional_features'] else None

            # Convert to double to get network to work
            sample = {'points': points[:, 0:3],  # XYZ
                      'normals': normals,
                      'features': features
                      }

            transform = CustomTransform(depth=self.cfg['octree_depth'], full_depth=self.cfg['full_depth'],
                                        use_normals=self.cfg['use_normals'])

            # Convert point cloud to an octree
            sample = transform(sample, idx=idx)

            # Add label and PlotID
            sample['target'] = torch.from_numpy(np.array(target)).float()
            sample['PlotID'] = PlotID

        else:

            # Return point cloud coords + features (points), biomass targets, and PlotID for one sample
            sample = {
                # Swap rows and cols of points tensor to fit required shape for most models
                'points': torch.from_numpy(points).float().permute(1, 0),
                'target': torch.from_numpy(np.array(target)).float(),
                'PlotID': PlotID
            }

        return sample


def augment_data(cfg, train_dataset, verbose=True):
    # Extract train filepaths from train dataset
    train_files = train_dataset.files

    # Augment pre-sampled training data
    if cfg['num_augs'] > 0:
        len_og_train_dataset = len(train_files)
        for i in range(cfg['num_augs']):
            # Augment data
            aug_trainset = PointCloudsInFilesPreSampled(cfg, set='train', partition=None, augment_data=True,
                                                        files=train_files)
            # Concat training and augmented training datasets
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])

        if verbose:
            print(
                f"Adding {cfg['num_augs']} augmentations of original {len_og_train_dataset} for a total of {len(train_dataset)} training samples.")

    return train_dataset
