import numpy as np
import nibabel as nb
import scipy.io as sio

import os
import h5py


def load(datafile, **kwargs):
    """
    Wrapper method to load common neuroimaging data.
    """

    assert os.path.exists(datafile)

    filename, file_extension = os.path.splitext(datafile)

    function_map = {'.mat': loadMat,
                    '.gii': loadGii}

    return function_map[file_extension](datafile, **kwargs)


def loadMat(infile, datasets=None):
    """
    Method to load .mat files.

    Parameters:
    - - - - -
        matFile : input .mat file
        datasets : if you know the specific key, provide key name.  Otherwise,
                    returns first non-private key data array.
    """

    try:
        matData = h5py.File(infile, mode='r')
    except OSError:
        try:
            matData = sio.loadmat(infile)
        except FileNotFoundError:
            err = 'Cannot read with h5py or scipy.io.'
            raise Warning(err)

    # if key name is known
    if datasets:
        try:
            mat = np.asarray(matData[datasets]).squeeze()
        except KeyError:
            pass
        else:
            if type(matData) == h5py._hl.files.File:
                mat = mat.T

    # otherwise, parse through keys, and select first non-private key name
    # and data array
    else:

        # remove private keys
        keys = [k for k in matData.keys() if k.startswith('_')]
        matData = {k: matData[k] for k in matData.keys() if k not in keys}

        # get first non-private key
        key = list(matData.keys())[0]
        mat = np.asarray(matData[key]).squeeze()

        if type(matData) == h5py._hl.files.File:
                mat = mat.T

    # if h5py, close object
    if type(matData) == h5py._hl.files.File:
        matData.close()

    return mat


def loadGii(infile, datasets=[], group=None):
    """
    Method to load Gifti files.

    Parameters:
    - - - - -
        infile : input gifti file
        darrayID : if array is .gii, often comes with multiple arrays
                    you can choose to specify which one
    """

    try:
        gii = nb.load(infile)
    except IOError:
        raise Warning('{} cannot be read.'.format(infile))

    if isinstance(datasets, int):
        datasets = [datasets]
    elif isinstance(datasets, np.ndarray):
        datasets = list(datasets)
    elif datasets == []:
        datasets = list(np.arange(len(gii.darrays)))

    if isinstance(gii, nb.gifti.GiftiImage):
        darray = []
        for j in datasets:
            darray.append(np.asarray(gii.darrays[j].data).squeeze())
        darray = np.column_stack(darray).squeeze()
    elif isinstance(gii, nb.nifti2.Nifti2Image):
        darray = np.asarray(gii.get_data()).squeeze()
    else:
        raise IOError('Cannot access array data.')

    return darray
