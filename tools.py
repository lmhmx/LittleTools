import datetime
import time
import sys
import os
import re
import torch
import numpy as np
import random
import numpy as np
import pickle
from scipy.io import savemat
class DataRecorder:
    def __init__(self, names=[], lengths=[], max_size=15, load_file = None) -> None:
        """
        names: the names of the variables
        lengths: the length of the variables
        max_size: the pre-allocated size of the variables
            - when it is a number, all variables allocate the same size
            - when it is a list, it assign different size for the variables  
        load_file: the file to be loaded
            - when it is None, do not load
            - when it is not None, excute self.load()
        """
        if(load_file is None):
            self.data = {}
            self.max_sizes = {}
            self.current_size = {}
            self.add_names(names, lengths, max_size)
        else:
            self.load(load_file)
    def add_names(self, names, lengths=[], max_size=15):
        """
        names: the names of the variables
        lengths: the length of the variables
        max_size: the pre-allocated size of the variables
            - when it is a number, all variables allocate the same size
            - when it is a list, it assign different size for the variables  
        """
        max_sizes = None
        if (type(max_size)==int):
            max_sizes = [max_size]*len(names)
        else:
            max_sizes = max_size
        for i in range(len(names)):
            if(lengths==[]):
                self.add_name(names[i], -1, max_sizes[i])
            else:
                self.add_name(names[i], lengths[i], max_sizes[i])
            
    def add_name(self, name, length=-1, max_size=15):
        """
        name: str
        length: int
        max_size: int
        """
        if(length!=-1):
            self.data[name]=np.zeros([max_size, length], dtype=np.float32)
            self.max_sizes[name] = max_size
            self.current_size[name] = 0

    def add(self, x: np.ndarray, name):
        """
        x: vector
        name: str
        """
        if(name not in self.data.keys()):
            self.add_name(name, length=len(x.flatten()), max_size=15)
        if(self.current_size[name]==self.max_sizes[name]):
            tmp = self.data[name]
            self.data[name] = np.zeros([np.shape(self.data[name])[0]*2, np.shape(self.data[name])[1]], dtype=np.float32)
            self.data[name][0:self.max_sizes[name], :] = tmp
            self.max_sizes[name] *= 2
            del tmp
        self.data[name][self.current_size[name], :] = x.flatten()
        self.current_size[name] += 1
    def get(self, name):
        return self.data[name][0:self.current_size[name], :]
    def clear(self):
        del self.data
        # for key in self.data.keys():
        #     del self.data[key]
    def save(self, path="./record/recorder.pkl"):
        with open(path,'wb') as file:
            pickle.dump({"data":self.data, 
                         "max_sizes":self.max_sizes,
                         "current_size":self.current_size},file)
    def load(self, path = "./record/recorder.pkl"):
        with open(path,'rb') as file:
            data = pickle.load(file)
            self.data=data["data"]
            self.max_sizes=data["max_sizes"]
            self.current_size = data["current_size"]
    def save_as_mat(self, path="./record/recorder.mat"):
        """
        This function works for outputing data accessed by MATLAB, which is only used when plotting.
        Be sure to not use it anytime else.
        """
        savemat(path, {k:self.data[k][0:self.current_size[k]] for k in self.data.keys()})

def time_int():
    if(not(hasattr(time_int, "is_init"))):
        time_int.is_init = True
        time_int.start_time = time.time()
    return int(time.time()-time_int.start_time)

def time_str():
    return datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')

def search_file_from_path(path:str, search_target):
    dirs = os.listdir(path)
    result = None
    for dir in dirs:
        # If we fine such a file
        if(re.match(search_target, dir) is not None):
            result = dir
            break
    else:
        # We cannot find such a file
        print("Warning: in the path '{}', there is not a file name started by '{}'".format(path, search_target))
        return ""
    return os.path.join(path, result)

class Print_Logger(object):
    def __init__(self, filename="./log/log"+time_str()+".txt",
                  show_num = 1,
                  show_in_terminal = True,
                  show_in_file = True):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.num = 0
        self.show_num = show_num
        self.show_in_terminal = show_in_terminal
        self.show_in_file = show_in_file
 
    def write(self, message):
        if(self.num%self.show_num==0):
            if(self.show_in_terminal):
                self.terminal.write(message)
            if(self.show_in_file):
                self.log.write(message)
        self.num += 1
    def flush(self):
        pass
class Counter:
    def __init__(self):
        self._num = 0
    @property
    def count(self):
        self._num+=1
        return self._num

def set_rand_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if(__name__=="__main__"):
    print("current time: {}".format(time_str()))
    
    print("Testing DataRecorder")
    data_recorder = DataRecorder()
    data_recorder.load()
