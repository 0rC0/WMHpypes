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
        desc="file to segment",
        mandatory=True,
        copyfile=False,
    )

    intlim = traits.Int(
                        100,
                        usedefault=True,
    )

    addnoise = traits.Float(0.5,
                            usedefault=True,
                            desc='strength of additional noise in noise-free regions')

    rician = traits.Int(0,
                         usedefault=True,
                         desc='use rician noise distribution')


class CAT12SANLMDenoisingOutputSpec(TraitedSpec):

    out_file = File(desc='out file')


class CAT12SANLMDenoising(SPMCommand):
    """
    Example:
    =======
    from wmhpypes.interfaces import cat12
    c = cat12.CAT12SANLMDenoising()
    c.inputs.in_files='sub-test_FLAIR.nii'
    c.inputs.rician = 0
    c.run()
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
        elif opt in ["intlim", "addnoise", "rician"]:
            return Cell2Str(val)

        return super(CAT12SANLMDenoising2, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(os.getcwd(), 'sanlm_' + self.inputs.in_files[0].split('/')[-1])
        return outputs

class Cell2Str(Cell):
    def __str__(self):
        """Convert input to appropriate format for cat12"""
        return "{'%s'}" % self.to_string()