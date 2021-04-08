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

from ..base import (
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
        position=-2,
    )

    out_file = File(
        desc="output file",
        genfile=True,
        argstr="%s",
        position=-1,
        name_source=["input_file"],
        hash_files=False,
        name_template="%s.mnc",
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
        desc="source Minc file", genfile=True, mandatory=True, argstr="%s"
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
            "Default value: 4"
        ),
        argstr="-mt %s",
    )
    # ToDo: a lot of options to be implemented


class MincNLMOutputSpec(TraitedSpec):
    out_file = File(desc="output mnc file", exists=True)


class MincNLM(CommandLine):
    input_spec = MincNLMInputSpec
    output_spec = MincNLMOutputSpec
    _cmd = 'mincnlm'