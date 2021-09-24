import numpy as np

class MPI():
    @property
    def FLOAT(self):
        return np.float32

    @property
    def COMM_WORLD(self):
        return self

    @staticmethod
    def Allgatherv(reward, reward_array):
        reward, dtype = reward
        reward_array, dtype = reward_array
        reward_array[...] = reward

    @staticmethod
    def Get_rank():
        return 0

    @staticmethod
    def Get_size():
        return 1
