import h5py

def load_hdf5_to_dict(file_path):
    data_dict = {}
    def recursively_extract(name, obj):
        if isinstance(obj, h5py.Dataset):
            data_dict[name] = obj[()]
    with h5py.File(file_path, 'r') as f:
        f.visititems(recursively_extract)
    return data_dict


