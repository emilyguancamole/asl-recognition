import ast
import torch

import pandas as pd
import torch.utils.data as torch_data

from random import randrange
from augmentations import *
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

HAND_IDENTIFIERS = [id + "R" for id in HAND_IDENTIFIERS] + [id + "L" for id in HAND_IDENTIFIERS]

def load_dataset(file_location: str):
    '''
    A function to convert a CSV file of videos by X & Y identifiers into 
    a list of data and labels
    :param file_location: the location of the CSV file
    :return: Tuple containing data and corresponding labels
    '''

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    labels = df["label"].to_list() # list of shape (videos,) of the corresponding labels for each element in data
    data = [] # list of shape (videos, frames, 55 identifiers, [x,y] coordinates)

    # add each row of the data frame to the data list
    # a row is a list of shape (frames, 55 identifiers, [x,y] coordinates)
    for row_index, row in df.iterrows():
        first_ft = BODY_IDENTIFIERS[0] + "_X"
        current_row = np.empty(shape=(len(ast.literal_eval(row[f'{first_ft}'])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2)) 
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])
        current_row = current_row.astype(np.float32)
        data.append(current_row)

    return data, labels # data: videos, frames, 55, 2


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:
    '''
    A function to convert a tensor of a video instance into a dictionary
    of landmark keys to a list of (frames, 2) representing the
    coordinates of the landmark as the video progresses by frame
    :param landmarks_tensor: a tensor representing a video instance of shape (frames, 55, 2)
    :return: a dictionary representing of {a str landmark : a list of shape (frames,2)}
    '''

    data_array = landmarks_tensor.numpy() # a tensor of shape (frames, 55, 2)
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index] # a list of shape (frames, 2)

    return output


def dictionary_to_tensor(landmarks_dict: dict) -> torch.Tensor:
    '''
    A function to convert a dictionary of landmark keys to a list of (frames, 2)
    into a tensor
    :param landmarks_dict: a dictionary representing of {a str landmark : a list of shape (frames,2)} 
    :return: a tensor representing a video instance of shape (frames, 55, 2)
    '''
    
    output = np.empty(shape=(len(landmarks_dict["eye_outerL"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


class LandmarkDataset(torch_data.Dataset):
    """
    Advanced object representation of the dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties
    """

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, transform=None, augmentations=False,
                augmentations_prob=0.5, normalize=True):
        """
        Initiates the dataset with the pre-loaded data from a file

        :param dataset_filename: a str path to the pre-loaded data file
        :param transform: any data transformation to be applied (default: None)
        :param augmentations: a boolean indicating whether to apply augmentations to data
        :param augmentatios_prob: a float indicating how frequently to apply augmentations to data
        :param normalize: a boolean indicating whether to apply normalization to data
        """

        data, labels = load_dataset(dataset_filename)

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.num_labels = len(np.unique(labels))
        self.transform = transform
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item (a video instance)
        at the desired index.

        :param idx: index of the item
        :return: a tuple containing both the depth map and the label
        """

        # a tensor of shape (frames, 55, 2) representing a video instance
        depth_map = torch.from_numpy(np.copy(self.data[idx])) 

        # label = torch.Tensor([self.labels[idx] - 1])
        label = torch.Tensor([self.labels[idx]])

        depth_map = tensor_to_dictionary(depth_map) 

        # Apply potential augmentations
        if self.augmentations and random.random() < self.augmentations_prob:

            selected_aug = randrange(4)

            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))

            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))

            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

            if selected_aug == 3:
                depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))

        if self.normalize: # Normalize the landmarks
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        depth_map = dictionary_to_tensor(depth_map)

        # Move the landmark position interval to improve performance
        depth_map = depth_map - 0.5

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass
