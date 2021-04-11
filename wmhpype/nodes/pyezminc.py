# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Minc command interfaces not included in the Nipype
"""
import glob
import os
import os.path
import re
import warnings

from nipype.interfaces.base import (
    TraitedSpec,
    CommandLineInputSpec,
    CommandLine,
    StdOutCommandLineInputSpec,
    StdOutCommandLine,
    File,
    Directory,
    InputMultiPath,
    OutputMultiPath,
    traits,
    isdefined,
)

class Nii2MNCInputSpec(CommandLineInputSpec):
    in_file = File(
        desc="input file for converting",
        exists=True,
        mandatory=True,
        argstr="%s",
    )

    out_file = File(
        os.path.join(os.getcwd(),'T1.mnc'),
        desc="output file",
        argstr="%s",
        usedefault=True,
    )

    quiet = traits.Bool(
        desc="Quiet operation",
        argstr="-quiet",
        usedefault=True,
        default_value=True,
    )


class Nii2MNCOutputSpec(TraitedSpec):
    out_file = File(desc='output file', exists=True)


class Nii2MNC(CommandLine):
    """
    Example
    -------
    ToDo
    """
    input_spec = Nii2MNCInputSpec
    output_spec = Nii2MNCOutputSpec
    _cmd = 'nii2mnc'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('_conv')
        return outputs

    def _gen_filename(self, suffix):
        return os.path.join(os.getcwd(), os.basename(self.inputs.out_file))



class BestLinRegS2InputSpec(CommandLineInputSpec):
    source = File(
        desc="source Minc file", exists=True, mandatory=True, argstr="%s", position=-4
    )

    target = File(
        desc="target Minc file", exists=True, mandatory=True, argstr="%s", position=-3
    )

    output_xfm = File(
        desc="output xfm file",
        genfile=True,
        argstr="%s",
        position=-2,
        name_source=["source"],
        hash_files=False,
        name_template="%s_bestlinreg.xfm",
        keep_extension=False,
    )

    output_mnc = File(
        desc="output mnc file",
        genfile=True,
        argstr="%s",
        position=-1,
        name_source=["source"],
        hash_files=False,
        name_template="%s_bestlinreg.mnc",
        keep_extension=False,
    )

    verbose = traits.Bool(
        desc="Print out log messages. Default: False.", argstr="-verbose"
    )
    clobber = traits.Bool(
        desc="Overwrite existing file.",
        argstr="-clobber",
        usedefault=True,
        default_value=True,
    )
    verbose = traits.Bool(
        desc="use 6-parameter transformation", argstr="-lsq6"
    )


    # FIXME Very bare implementation, none of these are done yet:
    """
    -init_xfm     initial transformation (default identity)
    -source_mask  source mask to use during fitting
    -target_mask  target mask to use during fitting
    -lsq9         use 9-parameter transformation (default)
    -lsq12        use 12-parameter transformation (default -lsq9)
    -lsq6         use 6-parameter transformation
    """


class BestLinRegS2OutputSpec(TraitedSpec):
    output_xfm = File(desc="output xfm file", exists=True)
    output_mnc = File(desc="output mnc file", exists=True)


class BestLinRegS2(CommandLine):
    """Hierachial linear fitting between two files.
    The bestlinreg script is part of the EZminc package:
    https://github.com/BIC-MNI/EZminc/blob/master/scripts/bestlinreg.pl
    Examples
    --------
    >>> from nipype.interfaces.minc import BestLinReg
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data
    >>> input_file = nonempty_minc_data(0)
    >>> target_file = nonempty_minc_data(1)
    >>> linreg = BestLinReg(source=input_file, target=target_file)
    >>> linreg.run() # doctest: +SKIP
    """

    input_spec = BestLinRegS2InputSpec
    output_spec = BestLinRegS2OutputSpec
    _cmd = "bestlinreg_s2"

class MincNLMInputSpec(CommandLineInputSpec):
    in_file = File(
        desc="source Minc file", exists=True, mandatory=True, argstr="%s"
    )

    out_file = File(
        desc="source Minc file", argstr="%s"
    )

    verbose = traits.Bool(
        desc="Print out log messages. Default: False.", argstr="-verbose"
    )
    clobber = traits.Bool(
        desc="Overwrite existing file.",
        argstr="-clobber",
        usedefault=True,
        default_value=True,
    )
    beta = traits.Float(
        desc=(
            "Beta Value"
            "Default value: 1"
        ),
        argstr="-beta %s",
    )
    threads = traits.Int(
        desc=(
            "Number of threads"
            "Default value: 1"
        ),
        argstr="-mt %s",
        usedefault=True,
        default_value=1
    )
    # ToDo: a lot of options to be implemented


class MincNLMOutputSpec(TraitedSpec):
    out_file = File(desc="output mnc file", exists=True)


class MincNLM(CommandLine):
    input_spec = MincNLMInputSpec
    output_spec = MincNLMOutputSpec
    _cmd = 'mincnlm'

class NUCorrectInputSpec(StdOutCommandLineInputSpec):
    # Implementation as freesurfer nu_correct
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        desc="input volume. Input can be any format accepted by mri_convert.",
    )
    # optional
    out_file = File(
        argstr="%s",
        name_source=["in_file"],
        name_template="%s_output",
        hash_files=False,
        keep_extension=True,
        desc="output volume. Output can be any format accepted by mri_convert. "
        + "If the output format is COR, then the directory must exist.",
    )
    iterations = traits.Int(
        4,
        usedefault=True,
        argstr="-iter %d",
        desc="Number of iterations to run nu_correct. Default is 4. This is the number of times "
        + "that nu_correct is repeated (ie, using the output from the previous run as the input for "
        + "the next). This is different than the -iterations option to nu_correct.",
    )
    distance = traits.Int(argstr="-distance %d",
                          desc="N3 -distance option")
    mask = File(
        exists=True,
        argstr="-mask %s",
        desc="brainmask volume. Input can be any format accepted by mri_convert.",
    )
    clobber = traits.Bool(
        desc="Overwrite existing file.",
        argstr="-clobber",
        usedefault=True,
        default_value=True,
    )

class NUCorrectOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output volume")

class NUCorrect(CommandLine):
    input_spec = NUCorrectInputSpec
    output_spec = NUCorrectOutputSpec
    _cmd = 'nu_correct'

class VolumePolInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        desc="Input volume",
    )
    template = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        desc="Template.",
    )
    order = traits.Int(
        default_value=1,
        usedefault=True,
        argstr="--order %d",
        desc="approximation order",
    )
    noclamp = traits.Bool(
        desc="don't clamp highlights.",
        argstr="--noclamp",
    )
    expfile = File(
        exists=True,
        mandatory=True,
        argstr="--expfile %s_T1_norm",
        desc="write output to the expression file ",
    )
    clobber = traits.Bool(
        desc="Overwrite existing file.",
        argstr="--clobber",
        usedefault=True,
        default_value=True,
    )


class VolumePolOutputSpec(TraitedSpec):
    out_file = File(desc="output file", exists=True)


class VolumePol(CommandLine):
    input_spec = VolumePolInputSpec
    output_spec = VolumePolOutputSpec
    _cmd = 'volume_pol'
