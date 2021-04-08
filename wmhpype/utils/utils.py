

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed


def nii2bool(nii, threshold=0.5):
    """
    Threshold a NifTI image by 0.5
    :returns bool numpy array. True if voxel value > 0.5
    """
    nii_arr = nib.load(nii).get_fdata()
    return np.where(nii_arr > threshold, 1, 0)


def dice(im1, im2):
    """
    Ref/Credits: https://gist.github.com/JDWarner/6730747
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def dsc_nii(nii1, nii2, threshold=0.5):
    return dice(nii2bool(nii1, threshold=threshold), nii2bool(nii2, threshold=threshold))


def dsc_nii_mp(list1, list2, n_jobs=None):
    return Parallel(n_jobs=n_jobs)(delayed(dsc_nii)(i, j) for i in list1 for j in list2)