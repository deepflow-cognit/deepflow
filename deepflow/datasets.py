if __name__ == "main":
    import os,numpy as np
class mnist:
    def __init__():
        try:
            os.system("wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
            print("\nDownloading mnist dataset from tf keras API: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
            
            
        except:
            print("\ncognit.deepflow - mnist dataset failed to download.")
    def load(path="./mnist.npz"):
        np.load(path)