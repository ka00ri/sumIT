import os
from pathlib import Path
from typing import List, Tuple
from itertools import product
from multiprocessing import Process

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset

# Set numpy seed
np.random.seed(40)


def make_unique_pairs(num_classes: int=10) -> List[Tuple[int, int]]:
    """ Takes in a number n, returns a list of unique pairs in range 0 to n-1. """

    return [pair for pair in product(np.arange(num_classes), np.arange(num_classes))]


def sample_unique_pairs(test_size: int, unique_pairs:  List[Tuple[int, int]]):
    """ Randomly splits a set of unique pairs into test and train subsets. """

    n = len(unique_pairs)
    test_dataset_indices = np.random.choice(np.arange(n), test_size, replace=False)
    train_dataset_indices = list(set(np.arange(n)) - set(test_dataset_indices))

    test_pairs = np.take(unique_pairs, test_dataset_indices, axis=0)
    train_pairs = np.take(unique_pairs, train_dataset_indices, axis=0)
    return train_pairs, test_pairs


def load_data(data_path: str):
    """ Loads MNIST datasets, in case not in data_path, and returns loaded train 
    and test datasets. 
    """

    trainset = torchvision.datasets.MNIST(data_path, train=True, 
                                        download=True)

    testset = torchvision.datasets.MNIST(data_path, train=False, 
                                        download=True)

    return trainset, testset


def save_image_pair(directory: str, image: Image.Image, name: str) -> None:
    """ Takes in an image and saves it under directory and name. """

    dir_path = os.path.join(os.path.abspath(''), directory)
    Path(dir_path).mkdir(parents=True, exist_ok=True)    
    image.save(os.path.join(dir_path, name))


def prepare_image_pairs(data: Dataset, 
                       mode: str, 
                       combinations: List[Tuple[int, int]], 
                       num_samples_per_combination: int, path: str) -> None:
    """ Takes in a tensorvision dataset, samples ramdomly images with labels corresponding 
        to a unique pair combination, then saves an image combining the two sampled 
        images. The process is repeated num_samples_per_combination times.

    data         : train or test MNIST images and targets dataset 
    mode         : save data under "train" or "test" subfolder
    combinations : unique pairs to sample images pairs for
    num_samples  : number of pairs to generate per class combination
    """
    
    images = data.data.numpy()
    targets = data.targets.numpy()

    print(f"Processing {mode} dataset...")
    
    for c_1, c_2 in combinations:
        print(f"Processing current image pair with labels: {c_1}, {c_2}")
        idx1 = np.random.choice(np.where(targets==c_1)[0], num_samples_per_combination)
        idx2 = np.random.choice(np.where(targets==c_2)[0], num_samples_per_combination)

        for i in range(num_samples_per_combination):
            pair_image = np.zeros((28,56), dtype="uint8")
            pair_image[:,:28] = images[idx1[i]]
            pair_image[:,28:] = images[idx2[i]]
            to_save_image = Image.fromarray(pair_image)
            to_save_name = f"{c_1}_{c_2}_{i}.png"
            to_save_subfolder = f"{mode}/{c_1}_{c_2}"
            to_save_path = os.path.join('', *[path, 'data', to_save_subfolder])
            save_image_pair(to_save_path, to_save_image, to_save_name)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Current absolute path: ", dir_path)

    # Load dataset via torchvision
    trainset, testset = load_data(dir_path)
    print(f"Loaded {len(trainset)} training samples and {len(testset)} testing samples")

    # 10 digit classes results into 100 unique pairs
    unique_pairs = make_unique_pairs(10) 
    print(f"Total of Unique pairs: \n", len(unique_pairs))
    assert len(unique_pairs) == 100

    # Split test and train/eval by unique pairs
    num_pairs_in_test = 15
    train_pairs, test_pairs = sample_unique_pairs(num_pairs_in_test, unique_pairs) 
    print(f"Sampled {len(train_pairs)*100/len(unique_pairs)}% unique pairs for training.")
    print(f"Sampled {len(test_pairs)*100/len(unique_pairs)}% unique pairs for testing.")

    # Number of samples to generate per unique pair
    samples_per_unique_pair = 100

    # Create images combining two MNIST images
    prepare_image_pairs(testset, "test", test_pairs, samples_per_unique_pair, dir_path)
    prepare_image_pairs(trainset, "train", train_pairs, samples_per_unique_pair, dir_path)

    print("Done writing data!")


    

