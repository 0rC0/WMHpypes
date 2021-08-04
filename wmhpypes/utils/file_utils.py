# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os


def add_suffix_to_filename(path, suffix):
    _, filename = os.path.split(path)
    basename = filename.split('.')[0]
    extensions = '.'.join(filename.split('.')[1:])
    return basename + suffix + extensions

