# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import os
import time
import numpy as np
import warnings
import scipy
import SimpleITK as sitk
from ..utils.utils import get_unet

import os
from nipype.interfaces.base import (
    TraitedSpec,
    File,
    traits,
    Directory,
    BaseInterfaceInputSpec,
    BaseInterface,
)

class PreprocessingInputSpec(BaseInterfaceInputSpec):

    t1w = File(exists=True,
               desc='Input T1w in an ITK readable format')

    flair = File(exists=True,
                 mandatory=True,
                 desc='Input FLAIR in an ITK readable format')

    rows_standard = traits.Int(
                        mandatory=True,
                        desc='input size for rows')

    cols_standard = traits.Int(
                        mandatory=True,
                        desc='input size for columns')

    thres = traits.Int(default_value=30,
                        usedefault=True,
                        desc='threshold')


class PreprocessingOutputSpec(TraitedSpec):

    preprocessed_array = traits.Array(
        desc='preprocessed images (FLAIR only or T1 and FLAIR) as Numpy array'
    )
    slice_shape = traits.Tuple(
        (traits.Int(), traits.Int(), traits.Int()),
        desc='slice shape'
    )

    # ToDo: implement save npz
    flair_array_npy = File(
        desc='preprocessed FLAIR as Numpy .npz'
    )
    #
    # two_modalities_npz = File(
    #     desc='preprocessed FLAIR and T1w concatenated as Numpy .npz'
    # )

class Preprocessing(BaseInterface):
    """
    Examples
    --------
    >>> from wmhpypes.interfaces import ibbmTum
    >>> preproc = ibbmTum.Preprocessing()
    >>> preproc.inputs.t1w = '../test_your_data_wmh/input_dir/T1.nii.gz'
    >>> preproc.inputs.flair = '../test_your_data_wmh/input_dir/FLAIR.nii.gz'
    >>> result = preproc.run()
    >>> result_array = result.outputs.preprocessed_array
    """

    input_spec = PreprocessingInputSpec
    output_spec = PreprocessingOutputSpec

    def preprocessing(self, FLAIR_array, T1_array = np.float32([])):
        # ToDo: FLAIR and T1_array should be attributes
        brain_mask = np.ndarray(np.shape(FLAIR_array)).astype(np.float32)
        brain_mask[FLAIR_array >= self.inputs.thres] = 1
        brain_mask[FLAIR_array < self.inputs.thres] = 0
        for iii in range(np.shape(FLAIR_array)[0]):
            brain_mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                brain_mask[iii, :, :])  # fill the holes inside brain

        FLAIR_array -= np.mean(FLAIR_array[brain_mask == 1])  # Gaussion Normalization
        FLAIR_array /= np.std(FLAIR_array[brain_mask == 1])

        num, rows_o, cols_o = np.shape(FLAIR_array)
        FLAIR_array = FLAIR_array[:,
                      int((rows_o - self.inputs.rows_standard) / 2):int((rows_o - self.inputs.rows_standard) / 2) + self.inputs.rows_standard,
                      int((cols_o - self.inputs.cols_standard) / 2):int((cols_o - self.inputs.cols_standard) / 2) + self.inputs.cols_standard]

        if self.inputs.t1w:
            T1_array -= np.mean(T1_array[brain_mask == 1])  # Gaussion Normalization
            T1_array /= np.std(T1_array[brain_mask == 1])
            T1_array = T1_array[:, int((rows_o - self.inputs.rows_standard) / 2):int((rows_o - self.inputs.rows_standard) / 2) + self.inputs.rows_standard,
                       int((cols_o - self.inputs.cols_standard) / 2):int((cols_o - self.inputs.cols_standard) / 2) + self.inputs.cols_standard]

            imgs_two_channels = np.concatenate((FLAIR_array[..., np.newaxis], T1_array[..., np.newaxis]), axis=3)
            return imgs_two_channels
        else:
            return FLAIR_array[..., np.newaxis]

    def _run_interface(self, runtime):

        FLAIR_array = sitk.GetArrayFromImage(sitk.ReadImage(self.inputs.flair))

        if self.inputs.t1w:
            T1_array = sitk.GetArrayFromImage(sitk.ReadImage(self.inputs.t1w))
            imgs_test = self.preprocessing(FLAIR_array, T1_array)
        else:
            imgs_test = self.preprocessing(FLAIR_array)
        out_arr_name = 'preprocessed.npy'
        np.save(os.path.abspath(out_arr_name), imgs_test)
        setattr(self, '_flair_array_npy', out_arr_name)
        setattr(self, '_preprocessed_array', imgs_test)
        setattr(self, '_slice_shape', imgs_test[0].shape)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['preprocessed_array'] = getattr(self, '_preprocessed_array')
        outputs['slice_shape'] = getattr(self, '_slice_shape')
        outputs['flair_array_npy'] = getattr(self, '_flair_array_npy')
        return outputs

class PostprocessingInputSpec(BaseInterfaceInputSpec):

    flair = File(exists=True,
                 mandatory=True,
                 desc='Input FLAIR in an ITK readable format')

    prediction = traits.Array(
        mandatory=True,
        desc='prediction to reshape'
    )

    rows_standard = traits.Int(
                        mandatory=True,
                        desc='input size for rows')

    cols_standard = traits.Int(
                        mandatory=True,
                        desc='input size for columns')

    per = traits.Float(
                        mandatory=True,
                        desc='ratio')


class PostprocessingOutputSpec(TraitedSpec):

    postprocessed_prediction = traits.Array(
        desc='get prediction in original shape'
    )

class Postprocessing(BaseInterface):

    input_spec = PostprocessingInputSpec
    output_spec = PostprocessingOutputSpec

    def postprocessing(self):
        FLAIR_array = sitk.GetArrayFromImage(sitk.ReadImage(self.inputs.flair))
        pred = self.inputs.prediction
        start_slice = int(np.shape(FLAIR_array)[0] * self.inputs.per)
        num_o, rows_o, cols_o = np.shape(FLAIR_array)  # original size

        original_pred = np.zeros(np.shape(FLAIR_array)).astype(np.float32)
        original_pred[:, int((rows_o - self.inputs.rows_standard) / 2):int((rows_o - self.inputs.rows_standard) / 2) + self.inputs.rows_standard,
        int((cols_o - self.inputs.cols_standard) / 2):int((cols_o - self.inputs.cols_standard) / 2) + self.inputs.cols_standard] = pred[:, :, :, 0]
        original_pred[0: start_slice, ...] = 0
        original_pred[(num_o - start_slice):num_o, ...] = 0
        return original_pred

    def _run_interface(self, runtime):
        orig_pred = self.postprocessing()
        np.save(os.path.join(os.getcwd(), 'orig_pred'), orig_pred)
        setattr(self, '_orig_pred', orig_pred)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['postprocessed_prediction']=getattr(self,'_orig_pred')
        return outputs

class PredictInputSpec(BaseInterfaceInputSpec):

    slice_shape = traits.Tuple(
        (traits.Int(), traits.Int(), traits.Int()),
        desc='slice shape'
    )

    preprocessed_array = traits.Array(
        mandatory=True,
        desc='Array from preprocessed FLAIR and T1w'
    )

    weights = traits.List(
        File(exists=True,),
        mandatory=True,
        desc='Weights as list of H5 files')

class PredictOutputSpec(TraitedSpec):

    prediction = traits.Array(
        desc='Prediction as array'
    )

    prediction_npy = traits.File(
        desc='Prediction as Numpy .npy'
    )

class Predict(BaseInterface):

    input_spec = PredictInputSpec
    output_spec = PredictOutputSpec

    def _run_interface(self, runtime):

        model = get_unet(self.inputs.slice_shape)
        p = list()
        for w in self.inputs.weights:
            model.load_weights(w)
            pred = model.predict(self.inputs.preprocessed_array, batch_size=1, verbose=1)
            p.append(pred)
        p_mean = np.array(p).mean(axis=0)

        out_arr_name = os.path.join(os.getcwd(), 'prediction.npy')
        np.save(out_arr_name, p_mean)
        setattr(self, '_prediction', p_mean)
        setattr(self, '_prediction_npy', out_arr_name)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['prediction'] = getattr(self, '_prediction')
        outputs['prediction_npy'] = getattr(self, '_prediction_npy')
        return outputs


class SavePredictionInputSpec(BaseInterfaceInputSpec):

    prediction_array = traits.Array(
        mandatory=True,
        desc='prediction as array'
    )
    output_filename = traits.Str(
        'prediction',
        usedefault=True,
        desc='output filename'
    )

class SavePredictionOutputSpec(TraitedSpec):

    prediction_nifti = traits.File(
        desc='prediction as NIfTI file'
    )

class SavePrediction(BaseInterface):

    input_spec = SavePredictionInputSpec
    output_spec = SavePredictionOutputSpec

    def _run_interface(self, runtime):

        out_path = os.path.join(os.getcwd(), self.inputs.output_filename + '.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(self.inputs.prediction_array), out_path)
        setattr(self, '_prediction_nii', out_path)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['prediction_nifti'] = getattr(self, '_prediction_nii')
        return outputs

class EnsembleInputSpec(BaseInterfaceInputSpec):

    in_arrays = traits.List(
        traits.Array(),
        mandatory=True,
        desc='Arrays list to esemble'
    )

class EnsembleOutputSpec(TraitedSpec):

    out_array = traits.Array(
        desc='averaged array'
    )

class Ensemble(BaseInterface):

    input_spec = EnsembleInputSpec
    output_spec = EnsembleOutputSpec

    def _run_interface(self, runtime):

        if len(self.inputs.in_arrays) > 1:
            out_arr = np.array(self.inputs.in_arrays).mean(axis=0)
        else:
            out_arr = self.inputs.in_arrays
        setattr(self, '_out_array', out_arr)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_array'] = getattr(self, '_out_array')
        return outputs


class ThresholdingInputSpec(BaseInterfaceInputSpec):
    in_array = traits.Array(
        mandatory=True,
        desc='input array'
    )
    thres = traits.Float(default_value=0.5,
                         usedefault=True,
                         desc='threshold')


class ThresholdingOutputSpec(TraitedSpec):
    out_array = traits.Array(
        desc='thresholded array'
    )


class Thresholding(BaseInterface):

    input_spec = ThresholdingInputSpec
    output_spec = ThresholdingOutputSpec

    def _run_interface(self, runtime):

        out_arr = np.where(self.inputs.in_array > self.inputs.thres, 1, 0)
        setattr(self, '_out_array', out_arr)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_array'] = getattr(self, '_out_array')
        return outputs

class TrainInputSpec(BaseInterfaceInputSpec):

    images = traits.Array(
        mandatory=True,
        desc='Images for the training as NumPy array'
    )

    masks = traits.Array(
        mandatory=True,
        desc='Masks for training as NumPy array'
    )

    model_path = traits.Directory(
        mandatory=True,
        desc='directory where to save the models'
    )

    ensemble_parameter = traits.Int(
        3,
        desc='ensemble parameter'
    )

    verbose = traits.Bool(
        True,
        usedefault=True,
        desc='Verbose'
    )

    batch_size = traits.Int(
        30,
        usedefault=True,
        desc='batch size, default 30'
    )

    epochs = traits.Int(
        5,
        usedefault=True,
        desc='epochs, default 5'
    )


class TrainOutputSpec(TraitedSpec):

    weights = traits.List(
        File(exists=True,),
        mandatory=True,
        desc='Weights as list of H5 files')


class Train(BaseInterface):

    input_spec = TrainInputSpec
    output_spec = TrainOutputSpec

    def train(self):

        model_name = 'model_'
        models_list = list()
        for iiii in range(self.inputs.ensemble_parameter):
            model_file=model_name + str(iiii) + '.h5'
            model = get_unet(img_shape)
            model_checkpoint = ModelCheckpoint(model_file,
                                               save_best_only=False,
                                               period = 2)
            model.fit(self.inputs.images,
                      self.inputs.masks,
                      batch_size=self.inputs.batch_size,
                      nb_epoch= self.inputs.epochs,
                      verbose=self.inputs.verbose,
                      shuffle=True,
                      validation_split=0.0,
                      callbacks=[model_checkpoint])
            model.save(model_file)
            models_list.append(os.path.join(os.getcwd(),model_file))
        return models_list

    def _run_interface(self, runtime):
        weights = train()
        setattr(self, '_weights', weights)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['weights'] = getattr(self, '_weights')
        return outputs
