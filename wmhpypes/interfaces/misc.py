# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import os
import time

import nibabel as nib
import numpy as np
import warnings
import scipy
import nibabel as nb
from nipype.interfaces.base import (
    TraitedSpec,
    File,
    traits,
    Directory,
    BaseInterfaceInputSpec,
    BaseInterface,
    InputMultiPath,
    OutputMultiPath,
    CommandLineInputSpec,
    CommandLine
)
from ..utils.file_utils import add_suffix_to_filename


class SaveNIfTIInputSpec(BaseInterfaceInputSpec):
    in_array = File(exists=True,
        desc='input array as NIfTI or NumPy array')

    in_header = File(
        exists=True,
        desc='NIfTI from that the header is taken'
    )

    in_matrix = File(exists=True,
             desc='NIfTI from that the matrix is taken'
    )

    out_filename = traits.Str(
        'out_nii',
        usedefault=True,
        desc='output file basename'
    )


class SaveNIfTIOutputSpec(TraitedSpec):
    out_file = File(desc='output NIfTI')


class SaveNIfTI(BaseInterface):

    input_spec = SaveNIfTIInputSpec
    output_spec = SaveNIfTIOutputSpec

    def _gen_output_name(self):
        return os.path.join(os.getcwd(), self.inputs.out_filename + '.nii.gz')

    def _run_interface(self, runtime):

        arr = nib.load(self.inputs.in_array).get_fdata()
        header = nib.load(self.inputs.in_header).header
        aff = nib.load(self.inputs.in_matrix).affine
        img = nib.Nifti2Image(arr,
                              header=header,
                              affine=aff)
        out_fname = self._gen_output_name()
        #print(type(img))
        #print(out_fname)
        nib.save(img, out_fname)
        setattr(self, '_out_file', out_fname)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = getattr(self, '_out_file')
        return outputs


class ExtractAffineInputSpec(BaseInterfaceInputSpec):

    in_nii = File(
        mandatory=True,
        exists=True,
        desc='input nifti. Its affine matrix will be returned'
    )

    output_filename = traits.Str(
        'affine.mat',
        usedefault=True,
        desc='output filename'
    )


class ExtractAffineOutputSpec(TraitedSpec):

    out_matrix = traits.Array(
        desc='The affine matrix as NumPy array'
    )

    out_file = File(
        desc='A text file containing the affine matrix'
    )


class ExtractAffine(BaseInterface):

    input_spec = ExtractAffineInputSpec
    output_spec = ExtractAffineOutputSpec

    def _gen_output_name(self):
        return os.path.join(os.getcwd(), self.inputs.output_filename)

    def _run_interface(self, runtime):
        affine = nib.load(self.inputs.in_nii).affine
        out_fname = self._gen_output_name()
        np.savetxt(out_fname, affine)
        setattr(self, '_out_matrix', affine)
        setattr(self, '_out_file', out_fname)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_matrix'] = getattr(self, '_out_matrix')
        outputs['out_file'] = getattr(self, '_out_file')
        return outputs

class GZipInputSpec(CommandLineInputSpec):
    input_file = File(desc="File", exists=True, mandatory=True, argstr="%s")


class GZipOutputSpec(TraitedSpec):
    output_file = File(desc = "Zip file", exists = True)


class GZip(CommandLine):
    input_spec = GZipInputSpec
    output_spec = GZipOutputSpec
    cmd = 'gzip'

    def _list_outputs(self):
            outputs = self.output_spec().get()
            outputs['output_file'] = os.path.abspath(self.inputs.input_file + ".gz")
            return outputs