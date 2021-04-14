# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import os.path
import re
import warnings
#import joblib
from sklearn.externals import joblib
import nibabel as nib
import numpy as np

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
    BaseInterfaceInputSpec,
    BaseInterface
)
"""
#[ID_Test, 
XFM_Files_Test, 
xfmf, 
Mask_Files_Test, 
maskf, 
T1_Files_Test, 
t1, 
T2_Files_Test, 
t2, 
PD_Files_Test, 
pd, 
FLAIR_Files_Test, 
flair, 
WMH_Files_Test, 
wmh,
cls_Files_Test, 
clsf] = get_addressess(path_Temp+'Preprocessed.csv')

"""


class PreTrainedInputSpec(BaseInterfaceInputSpec):
    id_test = traits.Str(
        mandatory=True,
        desc='participant ID',
    )
    classifier = traits.Str(
        'LDA',
        desc='classifier',
        usedefault=True,
    )
    xfm_file_test = traits.File(
        mandatory=True,
        desc='Transformation .xfm file',
    )
    mask_file_test = traits.File(
        mandatory=True,
        desc='Mask file test',
    )
    t1_file_test = traits.File(
        mandatory=True,
        exists=True,
        desc='T1 file test',
    )
    t2_file_test = traits.File(
        '',
        exists=True,
        desc='T2 file test',
    )
    pd_file_test = traits.File(
        '',
        exists=True,
        desc='PD File Test',
    )
    flair_file_test = traits.File(
        mandatory=True,
        exists=True,
        desc='FLAIR File test',
    )
    cls_file_test = traits.File(
        '',
        exists=True,
        desc='FLAIR File test',
    )
    path_trained_classifiers = traits.Directory(
        mandatory=True,
        exists=True,
        desc='Directory containing the trained classifier'
    )
    image_range = traits.Int(
        256,
        desc='Image rage',
        usedefault=True,
    )

class PreTrainedOutputSpec(TraitedSpec):
    out_nii = traits.File(
        desc="WMH mask NIfTI file",
        exists=True
    )
    # out_mnc = traits.File(
    #     desc="WMH mask MINC file",
    #     exists=True
    # )
    # out_jpg = traits.File(
    #     desc="WMH mask NIfTI file",
    #     exists=True
    # )


class PreTrained(BaseInterface):
    """
    Example:
    ========
    df = pd.read_csv('path-to-preprocessed-files/tmp/211428_Preprocessed.csv')
    c = classification.PreTrained()
    c.inputs.id_test = df.Subjects.iloc[0]
    c.inputs.xfm_file_test = df.XFMs.iloc[0]
    c.inputs.t1_file_test = df.T1s.iloc[0]
    c.inputs.mask_file_test = df.Masks.iloc[0]
    c.inputs.flair_file_test = df.FLAIRs.iloc[0]
    c.inputs.path_trained_classifiers = 'path-to-trained-classifier/Trained_Classifiers'
    c.inputs.classifier = 'RF'
    c.run()
    """
    input_spec = PreTrainedInputSpec
    output_spec = PreTrainedOutputSpec

    @staticmethod
    def _clean_name(s):
        return s.replace("[", '').replace("]", '').replace("'", '').replace(" ", '')

    def _classify_pt(self):
        id_test = self.inputs.id_test
        xfm_file_test = self.inputs.xfm_file_test
        t1_file_test = self.inputs.t1_file_test
        t2_file_test = self.inputs.t2_file_test
        pd_file_test = self.inputs.pd_file_test
        flair_file_test = self.inputs.flair_file_test
        cls_file_test = self.inputs.cls_file_test
        mask_file_test = self.inputs.mask_file_test
        classifier = str(self.inputs.classifier).rstrip().lstrip()
        print('classifier_{}'.format(classifier))
        path_trained_classifiers = self.inputs.path_trained_classifiers
        path_sp = os.path.join(self.inputs.path_trained_classifiers, 'SP.mnc')
        path_av_t1 = os.path.join(self.inputs.path_trained_classifiers, 'Av_T1.mnc')
        path_av_t2 = os.path.join(self.inputs.path_trained_classifiers, 'Av_T2.mnc')
        path_av_pd = os.path.join(self.inputs.path_trained_classifiers, 'Av_PD.mnc')
        path_av_flair = os.path.join(self.inputs.path_trained_classifiers, 'Av_FLAIR.mnc')
        image_range = self.inputs.image_range
        print('image_range: ', image_range)
        path_Temp = os.getcwd()
        path_output = os.getcwd()
        print('path_output ', os.getcwd())
        print('cls', classifier + ('_CLSexists' if cls_file_test else '_CLS'))
        training_file = classifier
        training_file += '_CLSexists' if cls_file_test else '_CLS'
        training_file += '_T1exists' if t1_file_test else '_T1'
        training_file += '_T2exists' if t2_file_test else '_T2'
        training_file += '_PDexists' if pd_file_test else '_PD'
        training_file += '_FLAIRexists' if flair_file_test else '_FLAIR'
        training_file += '.pkl'
        print('trainig_file ', training_file)
        path_saved_classifier = os.path.join(path_trained_classifiers, training_file)
        clf = self._conditional_import(classifier)
        print('path_saved_classifier', path_saved_classifier)
        clf = joblib.load(path_saved_classifier)
        if self.inputs.t1_file_test:
            T1_PDF_Healthy_Tissue = joblib.load(os.path.join(path_trained_classifiers, 'T1_HT.pkl'))
            T1_PDF_WMH = joblib.load(os.path.join(path_trained_classifiers, 'T1_WMH.pkl'))
        if self.inputs.t2_file_test:
            T2_PDF_Healthy_Tissue = joblib.load(os.path.join(path_trained_classifiers, 'T2_HT.pkl'))
            T2_PDF_WMH = joblib.load(os.path.join(path_trained_classifiers, 'T2_WMH.pkl'))
        if self.inputs.pd_file_test:
            PD_PDF_Healthy_Tissue = joblib.load(os.path.join(path_trained_classifiers, 'PD_HT.pkl'))
            PD_PDF_WMH = joblib.load(os.path.join(path_trained_classifiers, 'PD_WMH.pkl'))
        if self.inputs.flair_file_test:
            FLAIR_PDF_Healthy_Tissue = joblib.load(os.path.join(path_trained_classifiers, 'FLAIR_HT.pkl'))
            FLAIR_PDF_WMH = joblib.load(os.path.join(path_trained_classifiers, 'FLAIR_WMH.pkl'))
        #for i in range(0, len(id_test)):
        print('Segmenting Volumes: Subject: ID = ' + id_test)
        print('Mask: ' + mask_file_test)
        #Mask = minc.Image(mask_file_test).data
        Mask = nib.load(mask_file_test).get_fdata()
        print('Mask loades')
        ind_WM = (Mask > 0)
        if cls_file_test:
            #CLS = minc.Image(cls_file_test).data
            CLS = nib.load(mask_file_test).get_fdata()
            wm = (CLS == 3)
        new_command = 'export LD_LIBRARY_PATH=/home/anaconda3/envs/WMHpype/lib:$LD_LIBRARY_PATH &&  mincresample ' + path_sp + ' -like  ' + mask_file_test + ' -transform ' + xfm_file_test + ' -invert_transform ' + path_Temp + '_TT_tmp_sp.mnc -clobber'
        print('minresample')
        print(new_command)
        os.system(new_command)
        #spatial_prior = minc.Image(path_Temp + '_TT_tmp_sp.mnc').data
        spatial_prior = nib.load(path_Temp + '_TT_tmp_sp.mnc').get_fdata()

        FT = np.zeros(shape=(image_range, 1)).astype(float)
        if (t1_file_test):
            print('t1_file_test', t1_file_test)
            #T1 = minc.Image(t1_file_test).data
            print('nib.load.get_fdata() av_T1')
            T1 = nib.load(t1_file_test).get_fdata()
            new_command = 'export LD_LIBRARY_PATH=/home/anaconda3/envs/WMHpype/lib:$LD_LIBRARY_PATH && mincresample ' + path_av_t1 + ' -like ' + mask_file_test + ' -transform ' + xfm_file_test + ' -invert_transform ' + path_Temp + '_TT_tmp_t1.mnc -clobber'
            print('Executing {}'.format(new_command))
            os.system(new_command)
            #av_T1 = minc.Image(path_Temp + '_TT_tmp_t1.mnc').data
            print('nib.load.get_fdata() av_T1')
            av_T1 = nib.load(path_Temp + '_TT_tmp_t1.mnc').get_fdata()
            T1n = np.round(T1)
            for j in range(1, image_range):
                FT[j] = FT[j] + np.sum(T1n * Mask == j)
            print('np.sum(T1n * Mask =! 0)')
            print((T1n * Mask).min())
            print((T1n * Mask).max())
            print('T1n')
            print(T1n.sum())
            print(T1n.max())
            print(T1n.min())
            print('Mask')
            print(Mask.sum())
            print('image range')
            print(image_range)
            T1 = T1 * np.argmax(T1_PDF_Healthy_Tissue) / np.argmax(FT)
            T1[T1 < 1] = 1
            T1[T1 > (image_range - 1)] = (image_range - 1)
            T1_WM_probability = T1_PDF_Healthy_Tissue[np.round(T1[ind_WM]).astype(int)]
            T1_WMH_probability = T1_PDF_WMH[np.round(T1[ind_WM]).astype(int)]
            T1_WM_probability[T1[ind_WM] < 1] = 1
            T1_WMH_probability[T1[ind_WM] < 1] = 0
            N = len(T1_WMH_probability)
            X_t1 = np.zeros(shape=(N, 2))
            X_t1[0: N, 0] = T1[ind_WM]
            X_t1[0: N, 1] = av_T1[ind_WM]
            X_t1 = np.concatenate((X_t1, T1_WMH_probability, T1_WM_probability,
                                   (T1_WMH_probability + 0.0001) / (T1_WM_probability + 0.0001)), axis=1)

        if t2_file_test:
            #T2 = minc.Image(t2_file_test).data
            T2 = nib.load(t2_file_test).get_fdata()
            new_command = 'export LD_LIBRARY_PATH=/home/anaconda3/envs/WMHpype/lib:$LD_LIBRARY_PATH && mincresample ' + path_av_t2 + ' -like ' + mask_file_test + ' -transform ' + xfm_file_test + ' -invert_transform ' + path_Temp + '_TT_tmp_t2.mnc -clobber'
            os.system(new_command)
            #av_T2 = minc.Image(path_Temp + '_TT_tmp_t2.mnc').data
            av_T2 = nib.load(path_Temp + '_TT_tmp_t2.mnc').get_fdata()
            T2n = np.round(T2)
            for j in range(1, image_range):
                FT[j] = FT[j] + np.sum(T2n * Mask == j)
            T2 = T2 * np.argmax(T2_PDF_Healthy_Tissue) / np.argmax(FT)
            T2[T2 < 1] = 1
            T2[T2 > (image_range - 1)] = (image_range - 1)
            T2_WM_probability = T2_PDF_Healthy_Tissue[np.round(T2[ind_WM]).astype(int)]
            T2_WMH_probability = T2_PDF_WMH[np.round(T2[ind_WM]).astype(int)]
            T2_WM_probability[T2[ind_WM] < 1] = 1
            T2_WMH_probability[T2[ind_WM] < 1] = 0
            N = len(T2_WMH_probability)
            if t1_file_test:
                X_t2 = np.zeros(shape=(N, 3))
                X_t2[0: N, 0] = T2[ind_WM]
                X_t2[0: N, 1] = av_T2[ind_WM]
                X_t2[0: N, 2] = T2[ind_WM] / T1[ind_WM]
            else:
                X_t2 = np.zeros(shape=(N, 2))
                X_t2[0: N, 0] = T2[ind_WM]
                X_t2[0: N, 1] = av_T2[ind_WM]

            X_t2 = np.concatenate((X_t2, T2_WMH_probability, T2_WM_probability,
                                   (T2_WMH_probability + 0.0001) / (T2_WM_probability + 0.0001)), axis=1)

        if pd_file_test:
            #PD = minc.Image(pd_file_test).data
            PD = nib.load(pd_file_test).get_fdata()
            new_command = 'export LD_LIBRARY_PATH=/home/anaconda3/envs/WMHpype/lib:$LD_LIBRARY_PATH && mincresample ' + path_av_pd + ' -like ' + mask_file_test + ' -transform ' + xfm_file_test + ' -invert_transform ' + path_Temp + '_TT_tmp_pd.mnc -clobber'
            os.system(new_command)
            #av_PD = minc.Image(path_Temp + '_TT_tmp_pd.mnc').data
            av_PD = nib.load(path_Temp + '_TT_tmp_pd.mnc').get_fdata()
            PDn = np.round(PD)
            for j in range(1, image_range):
                FT[j] = FT[j] + np.sum(PDn * Mask == j)
            PD = PD * np.argmax(PD_PDF_Healthy_Tissue) / np.argmax(FT)
            PD[PD < 1] = 1
            PD[PD > (image_range - 1)] = (image_range - 1)
            PD_WM_probability = PD_PDF_Healthy_Tissue[np.round(PD[ind_WM]).astype(int)]
            PD_WMH_probability = PD_PDF_WMH[np.round(PD[ind_WM]).astype(int)]
            PD_WM_probability[PD[ind_WM] < 1] = 1
            PD_WMH_probability[PD[ind_WM] < 1] = 0
            N = len(PD_WMH_probability)
            if t1_file_test:
                X_pd = np.zeros(shape=(N, 3))
                X_pd[0: N, 0] = PD[ind_WM]
                X_pd[0: N, 1] = av_PD[ind_WM]
                X_pd[0: N, 2] = PD[ind_WM] / T1[ind_WM]
            else:
                X_pd = np.zeros(shape=(N, 2))
                X_pd[0: N, 0] = PD[ind_WM]
                X_pd[0: N, 1] = av_PD[ind_WM]
            X_pd = np.concatenate((X_pd, PD_WMH_probability, PD_WM_probability,
                                   (PD_WMH_probability + 0.0001) / (PD_WM_probability + 0.0001)), axis=1)

        if flair_file_test:
            #FLAIR = minc.Image(flair_file_test).data
            FLAIR = nib.load(flair_file_test).get_fdata()
            new_command = 'export LD_LIBRARY_PATH=/home/anaconda3/envs/WMHpype/lib:$LD_LIBRARY_PATH && mincresample ' + path_av_flair + ' -like ' + mask_file_test + ' -transform ' + xfm_file_test + ' -invert_transform ' + path_Temp + '_TT_tmp_flair.mnc -clobber'
            os.system(new_command)
            #av_FLAIR = minc.Image(path_Temp + '_TT_tmp_flair.mnc').data
            av_FLAIR = nib.load(path_Temp + '_TT_tmp_flair.mnc').get_fdata()
            FLAIRn = np.round(FLAIR)
            for j in range(1, image_range):
                FT[j] = FT[j] + np.sum(FLAIRn * Mask == j)
            FLAIR = FLAIR * np.argmax(FLAIR_PDF_Healthy_Tissue) / np.argmax(FT)
            FLAIR[FLAIR < 1] = 1
            FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
            FLAIR_WM_probability = FLAIR_PDF_Healthy_Tissue[np.round(FLAIR[ind_WM]).astype(int)]
            FLAIR_WMH_probability = FLAIR_PDF_WMH[np.round(FLAIR[ind_WM]).astype(int)]
            FLAIR_WM_probability[FLAIR[ind_WM] < 1] = 1
            FLAIR_WMH_probability[FLAIR[ind_WM] < 1] = 0
            N = len(FLAIR_WMH_probability)
            if t1_file_test:
                X_flair = np.zeros(shape=(N, 4))
                X_flair[0: N, 0] = FLAIR[ind_WM]
                X_flair[0: N, 1] = av_FLAIR[ind_WM]
                X_flair[0: N, 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                X_flair[0: N, 3] = FLAIR[ind_WM] / T1[ind_WM]
            else:
                X_flair = np.zeros(shape=(N, 3))
                X_flair[0: N, 0] = FLAIR[ind_WM]
                X_flair[0: N, 1] = av_FLAIR[ind_WM]
                X_flair[0: N, 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
            X_flair = np.concatenate((X_flair, FLAIR_WMH_probability, FLAIR_WM_probability,
                                      (FLAIR_WMH_probability + 0.0001) / (FLAIR_WM_probability + 0.0001)), axis=1)

        if cls_file_test:
            X = np.zeros(shape=(N, 2))
            X[0: N, 1] = wm[ind_WM] * spatial_prior[ind_WM]
        else:
            X = np.zeros(shape=(N, 1))
            X[0: N, 0] = spatial_prior[ind_WM]
        if t1_file_test:
            X = np.concatenate((X, X_t1), axis=1)
        if t2_file_test:
            X = np.concatenate((X, X_t2), axis=1)
        if pd_file_test:
            X = np.concatenate((X, X_pd), axis=1)
        if flair_file_test:
            X = np.concatenate((X, X_flair), axis=1)

        Y = np.zeros(shape=(N,))
        Binary_Output = clf.predict(X)
        Prob_Output = clf.predict_proba(X)
        #### Saving results #########################################################################################################################
        WMT_auto = np.zeros(shape=(len(Mask), len(Mask[0, :]), len(Mask[0, 0, :])))
        WMT_auto[ind_WM] = Binary_Output[0: N]

        #out = minc.Image(data=WMT_auto)
        mask_obj = nib.load(mask_file_test)
        out = nib.Nifti2Image(WMT_auto, header=mask_obj.header, affine=mask_obj.affine)
        str_WMHo = path_output + classifier + '_' + id_test
        #out.save(name=str_WMHo + '_WMH.mnc', imitate=mask_file_test)
        nib.save(out, str_WMHo + '_WMH.nii.gz')

        Prob_auto = np.zeros(shape=(len(Mask), len(Mask[0, :]), len(Mask[0, 0, :])))
        Prob_auto[ind_WM] = Prob_Output[0: N, 1]
        #out = minc.Image(data=Prob_auto)
        #out.save(name=str_WMHo + '_P.mnc', imitate=mask_file_test)
        prob_out = nib.Nifti2Image(Prob_auto, header=mask_obj.header, affine=mask_obj.affine)
        nib.save(prob_out, str_WMHo + '_P.nii.gz')

        # if t1_file_test:
        #     new_command = 'minc_qc.pl ' + t1_file_test + ' --mask ' + str_WMHo + '_WMH.mnc ' + str_WMHo + '_WMH.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
        #     os.system(new_command)
        # if t2_file_test:
        #     new_command = 'minc_qc.pl ' + t2_file_test + ' ' + str_WMHo + '_T2.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
        #     os.system(new_command)
        # if pd_file_test:
        #     new_command = 'minc_qc.pl ' + pd_file_test + ' ' + str_WMHo + '_PD.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
        #     os.system(new_command)
        # if flair_file_test:
        #     new_command = 'minc_qc.pl ' + flair_file_test + ' ' + str_WMHo + '_FLAIR.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
        #     os.system(new_command)

        # os.system('rm ' + path_Temp + '*')
        #setattr(self, '_out_mnc', str_WMHo + '_WMH.mnc')
        setattr(self, '_out_nii', str_WMHo + '_WMH.nii.gz')
        #setattr(self, '_out_jpg', str_WMHo + '_FLAIR.jpg')
        print('Segmentation Successfully Completed. ')

    def _run_interface(self, runtime):
        print('runtime')
        print(os.getcwd())
        self._classify_pt()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_jpg'] = getattr(self, '_out_jpg')
        outputs['out_mnc'] = getattr(self, '_out_mnc')
        outputs['out_nii'] = getattr(self, '_out_nii')