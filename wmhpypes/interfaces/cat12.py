# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.matlab import MatlabCommand
from string import Template
from nipype.interfaces.spm.base import (
    SPMCommandInputSpec,
    ImageFileSPM,
)

from nipype.utils.filemanip import split_filename

from nipype.interfaces.base import (
    InputMultiPath,
    TraitedSpec,
    traits,
    isdefined,
    File,
    Str,
)
from nipype.interfaces.spm.base import (
    SPMCommandInputSpec,
    ImageFileSPM,
    scans_for_fnames,
    scans_for_fname,
)
from nipype.interfaces.cat12.base import Cell


class CAT12SANLMDenoisingInputSpec(SPMCommandInputSpec):

    in_files = InputMultiPath(
        ImageFileSPM(exists=True),
        field="data",
        desc="Images for filtering.",
        mandatory=True,
        copyfile=False,
    )

    spm_type = traits.Enum(
        16,
        0,
        2,
        512,
        field='spm_type',
        usedefault=True,
        desc='Data type of the output images. 0 = same, 2 = uint8, 512 = uint16, 16 = single (32 bit)'

    )

    intlim = traits.Int(
        field='intlim',
        default_value=100,
        usedefault=True,
    )

    filename_prefix = traits.Str(
        field='prefix',
        default_value='sanlm_',
        usedefault=True,
        desc='Filename prefix. Specify  the  string  to be prepended to the filenames of the filtered image file(s). Default prefix is "samlm_".',
    )

    filename_suffix= traits.Str(
        field='suffix',
        default_value='',
        usedefault=True,
        desc='Filename suffix. Specify  the  string  to  be  appended  to the filenames of the filtered image file(s). Default suffix is "".'
    )

    addnoise = traits.Float(default_value=0.5,
                            usedefault=True,
                            field='addnoise',
                            desc='Strength of additional noise in noise-free regions. Add  minimal  amount  of noise in regions without any noise to avoid image segmentation problems. This parameter defines the strength of additional noise as percentage of the average signal intensity.')

    rician = traits.Enum(
        0,
        1,
        field='rician',
        usedefault=True,
        desc='''Rician noise
        MRIs  can  have  Gaussian  or  Rician  distributed  noise with uniform or nonuniform variance across the image. If SNR is high enough
        (>3)  noise  can  be  well  approximated by Gaussian noise in the foreground. However, for SENSE reconstruction or DTI data a Rician
        distribution is expected. Please note that the Rician noise estimation is sensitive for large signals in the neighbourhood and can lead to
        artefacts, e.g. cortex can be affected by very high values in the scalp or in blood vessels.''')

    replaceNANandINF = traits.Enum(
        1,
        0,
        field='replaceNANandINF',
        usedefault=True,
        desc='Replace NAN by 0, -INF by the minimum and INF by the maximum of the image.'
    )

    NCstr = traits.Enum(
        '-Inf',
        2,
        4,
        field='nlmfilter.optimized.NCstr',
        usedefault=True,
        desc='''Strength of Noise Corrections
        Strength  of  the  (sub-resolution)  spatial  adaptive    non local means (SANLM) noise correction. Please note that the filter strength is
        automatically  estimated.  Change this parameter only for specific conditions. The "light" option applies half of the filter strength of the
        adaptive  "medium"  cases,  whereas  the  "strong"  option  uses  the  full  filter  strength,  force sub-resolution filtering and applies an
        additional  iteration.  Sub-resolution  filtering  is  only  used  in  case  of  high image resolution below 0.8 mm or in case of the "strong"
        option. light = 2, medium = -Inf, strong = 4'''
    )


class CAT12SANLMDenoisingOutputSpec(TraitedSpec):

    out_file = File(desc='out file')


class CAT12SANLMDenoising(SPMCommand):
    """
    Spatially adaptive non-local means (SANLM) denoising filter

    This  function  applies  an spatial adaptive (sub-resolution) non-local means denoising filter
    to  the  data.  This  filter  will  remove  noise  while  preserving  edges. The filter strength is
    automatically estimated based on the standard deviation of the noise.

    This   filter   is  internally  used  in  the  segmentation  procedure  anyway.  Thus,  it  is  not
    necessary (and not recommended) to apply the filter before segmentation.


    Examples
    --------
    >>> from nipype.interfaces import cat12
    >>> c = cat12.CAT12SANLMDenoising()
    >>> c.inputs.in_files='sub-test_FLAIR.nii'
    >>> c.run()
    """

    input_spec = CAT12SANLMDenoisingInputSpec
    output_spec = CAT12SANLMDenoisingOutputSpec

    def __init__(self, **inputs):
        _local_version = SPMCommand().version
        if _local_version and "12." in _local_version:
            self._jobtype = "tools"
            self._jobname = "cat.tools.sanlm"

        SPMCommand.__init__(self, **inputs)

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == "in_files":
            if isinstance(val, list):
                return scans_for_fnames(val)
            else:
                return scans_for_fname(val)

        return super(CAT12SANLMDenoising, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        pth, base, ext = split_filename(self.inputs.in_files[0])
        outputs['out_file'] = os.path.join(os.getcwd(), self.inputs.filename_prefix +
                                           base +
                                           self.inputs.filename_suffix +
                                           ext)
        return outputs



class Cell2Str(Cell):
    def __str__(self):
        """Convert input to appropriate format for cat12"""
        return "{'%s'}" % self.to_string()