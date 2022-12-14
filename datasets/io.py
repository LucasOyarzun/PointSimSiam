import h5py
import numpy as np
import os


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".npy"]:
            return cls._read_npy(file_path)
        elif file_extension in [".h5"]:
            return cls._read_h5(file_path)
        elif file_extension in [".txt"]:
            return cls._read_txt(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, "r")
        return f["data"][()]
