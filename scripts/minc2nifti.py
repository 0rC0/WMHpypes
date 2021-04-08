"""
Convert minc to nifti format.
Credits:  @ofgulban https://gist.github.com/ofgulban/46d5c51ea010611cbb53123bb5906ca9
"""

import os
import numpy as np
from nibabel import load, save, Nifti1Image

minc = load("/path/to/file.mnc.gz")
basename = minc.get_filename().split(os.extsep, 1)[0]

affine = np.array([[0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])

out = Nifti1Image(minc.get_data(), affine=affine)
save(out, basename + '.nii.gz')