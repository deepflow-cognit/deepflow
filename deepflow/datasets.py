
import os,numpy as np
import requests,gzip

class mnist:
    def __init__(url):
        pass
    
    def download(url):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for failed downloads
            with open('./mnist.npz', "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            
        except:
            print("\ndeepflow.datasets.mnist() - mnist dataset failed to download.")
            
    def load(dataset_dir="mnist_data"):
        """Downloads and loads the MNIST dataset (training and testing images and labels) as NumPy arrays.

        Args:
            dataset_dir (str, optional): The directory to download and store the MNIST dataset files.
                Defaults to "mnist_data".

        Returns:
            tuple: A tuple of four NumPy arrays containing the training images,
                training labels, testing images, and testing labels.
        """

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)  # Create the directory if it doesn't exist

        print("Downloading from source: http://yann.lecun.com/exdb/mnist/")
        base_url = "http://yann.lecun.com/exdb/mnist/"
        print("Downloading: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
        train_images_url = f"{base_url}train-images-idx3-ubyte.gz"
        print("Downloading: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
        train_labels_url = f"{base_url}train-labels-idx1-ubyte.gz"
        print("Downloading: http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
        test_images_url = f"{base_url}t10k-images-idx3-ubyte.gz"
        print("Downloading: http://yann.lecun.com/exdb/mnist.download/t10k-labels-idx1-ubyte.gz")
        test_labels_url = f"{base_url}t10k-labels-idx1-ubyte.gz"

        # Download files if they don't exist
        mnist.download(train_images_url, os.path.join(dataset_dir, "train_images.gz"))
        mnist.download(train_labels_url, os.path.join(dataset_dir, "train_labels.gz"))
        mnist.download(test_images_url, os.path.join(dataset_dir, "t10k_images.gz"))
        mnist.download(test_labels_url, os.path.join(dataset_dir, "t10k_labels.gz"))

        # Load data from downloaded files
        with gzip.open(os.path.join(dataset_dir, "train_images.gz"), 'rb') as f:
            train_images = np.frombuffer(f.read(), np.uint8, offset=16)
            train_images = train_images.reshape(-1, 28, 28)

        with gzip.open(os.path.join(dataset_dir, "train_labels.gz"), 'rb') as f:
            train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(dataset_dir, "t10k_images.gz"), 'rb') as f:
            test_images = np.frombuffer(f.read(), np.uint8, offset=16)
            test_images = test_images.reshape(-1, 28, 28)

        with gzip.open(os.path.join(dataset_dir, "t10k_labels.gz"), 'rb') as f:
            test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        return train_images, train_labels, test_images, test_labels
    
    
