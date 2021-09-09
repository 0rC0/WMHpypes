# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

# WMH Segmentation workflow
# ==========================
#
# Exmple Usage:
# =============
# from wmhpypes.workflows import ibbmTum_wf
# from nipype.pipeline.engine import Workflow, Node

# from nipype import IdentityInterface
# test_wf = ibbmTum_wf.get_test_wf()
# flair = Node(interface=IdentityInterface(fields=['flair']), name='flair')
# flair.inputs.flair = os.path.abspath('./sub-test_FLAIR.nii')
# weights = Node(interface=IdentityInterface(fields=['weights']), name='weights')
# weights.inputs.weights = [os.path.abspath('./0.h5')]
# sink = Node(interface=DataSink(), name ='sink')
# sink.inputs.base_directory = './out'
#
# wmh = Workflow(name='wmh', base_dir='./wf')
# wmh.connect(weights, 'weights', test_wf, 'inputspec.weights')
# wmh.connect(flair, 'flair', test_wf, 'inputspec.flair')
# wmh.connect(test_wf, 'outputspec.prediction_nifti', sink, '@pred')
# wmh.run()

from nipype.pipeline.engine import Workflow, Node
from nipype import DataGrabber, DataSink, IdentityInterface, MapNode, JoinNode
from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.utility import Function, Merge
import os
from ..interfaces.ibbmTum import Preprocessing, Predict, Postprocessing, SavePrediction, Ensemble, Thresholding


def get_test_wf(row_st=200,
                cols_st=200,
                thres_mask=30,
                per=0.125,
                thres_pred=0.5,
                cores=os.cpu_count()):

    inputspec = Node(interface=IdentityInterface(fields=['t1w', 'flair', 'weights']), name='inputspec')
    preproc = Node(interface=Preprocessing(rows_standard=row_st,
                                           cols_standard=cols_st,
                                           thres=thres_mask), name='preprocessing')
    predict = Node(interface=Predict(), name='predict')
    predict.interface._nprocs = cores
    predict.interface._memgb = 1
    thresholding = Node(interface=Thresholding(thres=thres_pred), name='thresholding')
    postproc = Node(interface=Postprocessing(rows_standard=row_st,
                                             cols_standard=cols_st,
                                             per=per), name='postprocessing')
    save = Node(interface=SavePrediction(output_filename='prediction'), name='save_prediction')
    outputspec = Node(interface=IdentityInterface(fields=['wmh_mask']), name='outputspec')

    test_wf = Workflow(name='ibbmTum_test_wf')
    test_wf.connect(inputspec, 't1w', preproc, 't1w')
    test_wf.connect(inputspec, 'flair', preproc, 'flair')
    test_wf.connect(inputspec, 'weights', predict, 'weights')
    test_wf.connect(preproc, 'preprocessed_array', predict, 'preprocessed_array')
    test_wf.connect(preproc, 'slice_shape', predict, 'slice_shape')
    test_wf.connect(predict, 'prediction', thresholding, 'in_array')
    test_wf.connect(thresholding, 'out_array', postproc, 'prediction')
    test_wf.connect(inputspec, 'flair', postproc, 'flair')
    test_wf.connect(postproc, 'postprocessed_prediction', save, 'prediction_array')
    test_wf.connect(save, 'prediction_nifti', outputspec, 'wmh_mask')

    return test_wf
