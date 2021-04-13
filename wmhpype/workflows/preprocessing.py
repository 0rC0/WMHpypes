# -*- coding: utf-8 -*-

from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Merge
from nipype import DataGrabber, DataSink, IdentityInterface
from nipype.interfaces import minc as minc
from ..nodes.pyezminc import Nii2MNC, BestLinRegS2, MincNLM, NUCorrect, VolumePol, NLfits
import os
from datetime import datetime


class WMHpreproc:

    def __init__(self, participants, inputs, templates, temp_dir, bids_root):
        self.participants = participants
        self.inputs = inputs
        self.templates = templates
        self.temp_dir = temp_dir
        self.bids_root = bids_root

    @staticmethod
    def gen_timestamp():
        return int(datetime.now().timestamp())

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

        t1_preproc = self.make_std_preprocessing_wf(templates_source, 't1')
        flair_preproc = self.make_std_preprocessing_wf(templates_source, 'flair')

        sink = Node(interface=DataSink(), name='sink')
        sinkdir = os.path.join(self.bids_root, 'derivatives', 'WMH')
        if not os.path.isdir(sinkdir):
            os.makedirs(sinkdir)
        sink.inputs.base_directory = sinkdir

        bestlinereg_s2 = Node(interface=BestLinRegS2(out_xfm='T1toMain.xfm',
                                                     clobber=True,
                                                     lsq6=True), name='bestlinreg_s2')
        resample = Node(interface=minc.Resample(), name='resample')

        wmhpreproc = Workflow(name='WMHpreprocess', base_dir=self.temp_dir)
        #wmhpreproc.connect(participant_source, 'participant', data_grabber, 'participant_id')
        wmhpreproc.connect(data_grabber, 't1', t1_preproc, 'nii2mnc.in_file')
        wmhpreproc.connect(data_grabber, 'flair', flair_preproc, 'nii2mnc.in_file')
        wmhpreproc.connect(t1_preproc, 'minccalc.output_file', sink, '@t1_vp')
        wmhpreproc.connect(flair_preproc, 'minccalc.output_file', sink, '@flair_vp')

        wmhpreproc.connect(flair_preproc, 'minccalc.output_file', bestlinereg_s2, 'target')
        wmhpreproc.connect(t1_preproc, 'minccalc.output_file', bestlinereg_s2,'source')
        wmhpreproc.connect(t1_preproc, 'resample.output_file', bestlinereg_s2, 'source_mask')
        wmhpreproc.connect(t1_preproc, 'minccalc.output_file', resample, 'input_file')
        wmhpreproc.connect(bestlinereg_s2, 'output_xfm', resample, 'transform')
        wmhpreproc.connect(flair_preproc, 'minccalc.output_file', resample, 'like')

        return wmhpreproc

    def make_std_preprocessing_wf(self, templates_source, modality):

        nii2mnc = Node(interface=Nii2MNC(out_file='{}_{}_tominc.mnc'.format(self.gen_timestamp(), modality)), name='nii2mnc')
        bestlinereg_s2 = Node(interface=BestLinRegS2(), name='bestlinereg_s2')
        resample = Node(interface=minc.Resample(nearest_neighbour_interpolation=True,
                                                clobber=True,
                                                invert_transformation=True),
                        name='resample')
        mincnlm = Node(interface=MincNLM(beta=0.7, out_file='{}_{}_NLM.mnc'.format(self.gen_timestamp(), modality)), name='minclnm')
        nu_correct = Node(interface=NUCorrect(iterations=200,
                                              distance=50,
                                              clobber=True), name='nu_correct')
        volume_pol = Node(interface=VolumePol(order=1,
                                              noclamp=True,
                                              clobber=True,
                                              expfile='{}_{}_norm.mnc'.format(self.gen_timestamp(), modality)), name='volume_pol')
        merge_for_calc = Node(interface=Merge(2), name='merge_for_calc')
        minccalc = Node(interface=minc.Calc(), name='minccalc')


        std_preproc = Workflow(name='WMH_{}_preproc'.format(modality), base_dir=self.temp_dir)
        std_preproc.connect(nii2mnc, 'out_file', bestlinereg_s2, 'source')
        std_preproc.connect(templates_source, '{}'.format(modality), bestlinereg_s2, 'target')
        std_preproc.connect(templates_source, 'mask', resample, 'input_file')
        std_preproc.connect(nii2mnc, 'out_file', resample, 'like')
        std_preproc.connect(bestlinereg_s2, 'output_xfm', resample, 'transformation')
        std_preproc.connect(nii2mnc, 'out_file', mincnlm, 'in_file')
        std_preproc.connect(mincnlm, 'out_file', nu_correct, 'in_file')
        std_preproc.connect(resample, 'output_file', nu_correct, 'mask')
        std_preproc.connect(nu_correct, 'out_file', volume_pol, 'in_file')
        std_preproc.connect(templates_source, '{}'.format(modality), volume_pol, 'template')
        std_preproc.connect(volume_pol, 'expfile', minccalc, 'expfile')
        std_preproc.connect(nu_correct, 'out_file', minccalc, 'input_files')

        if 't1' in modality.lowercase():

            bestlinereg_s2_2template = Node(interface=BestLinRegS2(), name='bestlinereg_s2_2template')
            resample_2template = Node(interface=minc.Resample(clobber=True), name='resample_2template')
            nlfits = Node(interface=NLfits(level=2, clobber=True,
                                           out_xfm='{}_T1toTemplate_pp_nlin.xfm'.format(self.gen_timestamp())), name='nl_fits')
            xfmconcat = Node(interface=minc.XfmConcat(), name='xfmconcat')

            std_preproc.connect(minccalc, 'output_file', bestlinereg_s2_2template, 'source')
            std_preproc.connect(templates_source, '{}'.format(modality), bestlinereg_s2_2template, 'target')
            std_preproc.connect(bestlinereg_s2, 'output_xfm', resample_2template, 'transformation')
            std_preproc.connect(templates_source, '{}'.format(modality), resample_2template, 'like')
            std_preproc.connect(minccalc, 'output_file', resample_2template, 'input_file')
            std_preproc.connect(resample_2template, 'output_file', nlfits, 'source')
            std_preproc.connect(templates_source, '{}'.format(modality), nlfits, 'target')
            std_preproc.connect(nlfits, 'out_xfm', xfmconcat, 'input_files')
            std_preproc.connect(bestlinereg_s2_2template, 'output_xfm', xfmconcat, 'input_files')

        return std_preproc
