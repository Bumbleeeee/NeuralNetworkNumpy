import requests
import os
import gzip
import numpy as np

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

def get_files():
    for fname in data_sources.values():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print("Downloading file: " + fname)
            resp = requests.get(base_url + fname, stream=True)
            resp.raise_for_status()  # Ensure download was succesful
            with open(fpath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=128):
                    fh.write(chunk)


def get_data_as_arrays():
    mnist_dataset = {}

    # Images
    for key in ("training_images", "test_images"):
        with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
            mnist_dataset[key] = np.frombuffer(
                mnist_file.read(), np.uint8, offset=16
            ).reshape(-1, 28 * 28)
    # Labels
    for key in ("training_labels", "test_labels"):
        with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
            mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

    x_train, y_train, x_test, y_test = (
        mnist_dataset["training_images"],
        mnist_dataset["training_labels"],
        mnist_dataset["test_images"],
        mnist_dataset["test_labels"]
    )
    return x_train, y_train, x_test, y_test

def normalize(arr):
    # normalize values to be [0, 1]
    return arr.astype(np.float64) / 255

# one hot encoding for labels
# broadcasting works b/c 10 and 1 fit together (== gives True/False then convert to floats)
def one_hot_encode(arr):
    return (arr.reshape(-1, 1) == np.arange(10)).astype(np.float64)


def get_data():
    get_files()

    x_train, y_train, x_test, y_test = get_data_as_arrays()

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test



