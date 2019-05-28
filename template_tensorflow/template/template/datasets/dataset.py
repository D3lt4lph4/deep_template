import tensorflow as tf

from tf.data import Dataset

class DatasetMNIST(Dataset):


    def init(self):
        pass
    
    def __iter__(self):
        raise NotImplementedError

    def apply(self, transformation_function):
        raise NotImplementedError
    
    def batch(self, batch_size, drop_remainder=False):
        raise NotImplementedError
    
    def cache(self, filename=""):
        raise NotImplementedError
    
    def concatenate(self, dataset):
        raise NotImplementedError
    
    def filter(self, predicate):
        raise NotImplementedError
    
    def flat_map(self, map_function):
        raise NotImplementedError
    
    

    def input_fn(self, data_file, num_epochs, shuffle, batch_size):
        raise NotImplementedError

    