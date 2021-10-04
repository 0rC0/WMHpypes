import sys, os, importlib

try:
    from wmhpypes.interfaces import ibbmTum
    from wmhpypes.workflows import ibbmTum_wf
    from wmhpypes.interfaces import cat12
except:
    sys.path.append(os.path.abspath('../'))
    from wmhpypes.interfaces import ibbmTum
    from wmhpypes.workflows import ibbmTum_wf
    from wmhpypes.interfaces import cat12

from nipype import (DataGrabber,
                    DataSink,
                    IdentityInterface,
                    MapNode,
                    Workflow,
                    Node)
from nipype.interfaces import fsl
from nipype.interfaces import spm
from nipype.algorithms.misc import Gunzip


