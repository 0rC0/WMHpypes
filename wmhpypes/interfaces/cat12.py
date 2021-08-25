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


class CAT12SANLMDenoisingInputSpec(SPMCommandInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc='Input file')

    v = traits.Int(3,
                   usedefault=True,
                   desc = 'size of search volume (M in paper)')

    f = traits.Int(1,
                   usedefault=True,
                   desc = 'size of neighborhood (d in paper)')

    rician = traits.Bool(0,
                         usedefault=True,
                         desc='use rician noise distribution')


class CAT12SANLMDenoisingOutputSpec(TraitedSpec):

    out_file = File(desc='out file')


class CAT12SANLMDenoising(SPMCommand):

    input_spec = CAT12SANLMDenoisingInputSpec
    output_spec = CAT12SANLMDenoisingOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_file=self.inputs.in_file,
                 v=self.inputs.v,
                 f=self.inputs.f,
                 rician=self.inputs.rician)
        script = Template("""in_file= '$in_file';
                             v = '$v';
                             f = '$f';
                             rician = '$rician';
                             cat_sanlm(in_file, v, f, rician);
        """)
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(os.getcwd(), 'sanlm_' + self.inputs.in_file)
        return outputs
