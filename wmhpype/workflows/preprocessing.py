# -*- coding: utf-8 -*-

from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Merge
from nipype import DataGrabber, DataSink, IdentityInterface
from nipype.interfaces import minc as minc
from ..nodes.pyezminc import Nii2MNC, BestLinRegS2, MincNLM, NUCorrect, VolumePol
import os

class WMHpreproc:

    def __init__(self, participants, inputs, templates, temp_dir, bids_root):
        self.participants = participants
        self.inputs = inputs
        self.templates = templates
        self.temp_dir = temp_dir
        self.bids_root = bids_root

    def make_wmh_preprocessing_wf(self):

        participant_source = Node(interface=IdentityInterface(fields=['participant_id']), name="participants_source")
        participant_source.iterables = ('participant', self.participants)

        templates_source = Node(interface=IdentityInterface(fields=list(self.templates.keys())), name="templates_source")
        templates_source.iterables = ('participant', self.participants)

        templates_source= Node(
            interface=DataGrabber(
                outfields=list(self.templates.keys())),
            name='templates_source')
        templates_source.inputs.base_directory = self.bids_root
        templates_source.inputs.template = '*'
        templates_source.inputs.field_template = self.templates
        templates_source.inputs.sort_filelist = True

        data_grabber = Node(
            interface=DataGrabber(
                outfields=list(self.inputs.keys())),
            name='data_grabber')
        data_grabber.inputs.base_directory = self.bids_root
        data_grabber.inputs.template = '*'
        data_grabber.inputs.field_template = self.inputs
        data_grabber.inputs.sort_filelist = True

        t1_preproc = self.make_t1_preprocessing_wf(templates_source)

        wmhpreproc = Workflow(name='WMHpreprocess', base_dir=self.temp_dir)
        #wmhpreproc.connect(participant_source, 'participant', data_grabber, 'participant_id')
        wmhpreproc.connect(data_grabber, 't1', t1_preproc, 'nii2mnc.in_file')
        #wmhpreproc.connect(data_grabber, 't1', t1_preproc, 'nii2mnc.out_file')
        return wmhpreproc

    def make_t1_preprocessing_wf(self, templates_source):

        nii2mnc = Node(interface=Nii2MNC(), name='nii2mnc')
        bestlinereg_s2 = Node(interface=BestLinRegS2(), name='bestlinereg_s2')
        resample = Node(interface=minc.Resample(nearest_neighbour_interpolation=True,
                                                clobber=True,
                                                invert_transformation=True),
                        name='resample')
        mincnlm = Node(interface=MincNLM(beta=0.7), name='minclnm')
        nu_correct = Node(interface=NUCorrect(iterations=200,
                                              distance=50,
                                              clobber=True), name='nu_correct')
        volume_pol = Node(interface=VolumePol(order=1,
                                              noclamp=True,
                                              clobber=True), name='volume_pol')
        merge_for_calc = Node(interface=Merge(2), name='merge_for_calc')
        minccalc = Node(interface=minc.Calc(), name='minccalc')


        wmht1preproc = Workflow(name='WMH_T1_preprocessing', base_dir=self.temp_dir)
        wmht1preproc.connect(nii2mnc, 'out_file', bestlinereg_s2, 'source')
        wmht1preproc.connect(templates_source, 't1', bestlinereg_s2, 'target')
        wmht1preproc.connect(templates_source, 'mask', resample, 'input_file')
        wmht1preproc.connect(nii2mnc, 'out_file', resample, 'like')
        wmht1preproc.connect(bestlinereg_s2, 'output_xfm', resample, 'transformation')
        wmht1preproc.connect(nii2mnc, 'out_file', mincnlm, 'in_file')
        wmht1preproc.connect(mincnlm, 'out_file', nu_correct, 'in_file')
        wmht1preproc.connect(resample, 'output_file', nu_correct, 'mask')
        wmht1preproc.connect(nu_correct, 'out_file', volume_pol, 'in_file')
        wmht1preproc.connect(templates_source, 't1', volume_pol, 'template')
        wmht1preproc.connect(volume_pol, 'out_file', minccalc, 'expfile')
        wmht1preproc.connect(nu_correct, 'out_file', minccalc, 'input_files')
        return wmht1preproc
