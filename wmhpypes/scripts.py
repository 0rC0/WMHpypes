# -*- coding: utf-8 -*-
import os

def doPreprocessing_T1(path_nlin_mask,path_Temp, ID_Test, WMH_Files_Test , wmh, T1_Files_Test , t1 , T2_Files_Test , t2 , PD_Files_Test , pd , FLAIR_Files_Test , flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair):
    nlmf = 'Y'
    nuf = 'Y'
    volpolf = 'Y'
    if '.nii' in T1_Files_Test[0]:
        fileFormat = 'nii'
    else:
        fileFormat = 'mnc'
    preprocessed_list = {}
    str_t1_proc = ''
    str_t2_proc = ''
    str_pd_proc = ''
    str_flair_proc = ''
    preprocessed_list_address = path_Temp + 'Preprocessed.csv'
    print('Preprocessing Images')
    for i in range(0, len(T1_Files_Test)):
        if (t1 != ''):
            str_File_t1 = str(T1_Files_Test[i]).replace("[", '').replace("]", '').replace("'", '').replace(" ", '')
            if (fileFormat == 'nii'):
                new_command = 'nii2mnc ' + str_File_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1.mnc'
            else:
                new_command = 'cp ' + str_File_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1.mnc'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = '/bestlinreg_s2 ' + path_Temp + str(
                ID_Test[i]) + '_T1.mnc ' + path_av_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'mincresample ' + path_nlin_mask + ' -transform ' + path_Temp + str(
                ID_Test[i]) + '_T1toTemplate.xfm' + ' ' + path_Temp + str(
                ID_Test[i]) + '_T1_Mask.mnc -invert_transform -like ' + path_Temp + str(
                ID_Test[i]) + '_T1.mnc -nearest -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1.mnc'
            str_main_modality = str_t1_proc
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_T1.mnc ' + path_Temp + str(
                    ID_Test[i]) + '_T1_NLM.mnc -beta 0.7 -clobber'
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc'
                str_main_modality = str_t1_proc
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc ' + path_Temp + str(
                    ID_Test[i]) + '_T1_N3.mnc -iter 200 -distance 50 -clobber' - mask
                '+ path_Temp + str(ID_Test[i]) + '
                _T1_Mask.mnc
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_N3.mnc'
                str_main_modality = str_t1_proc
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(
                    ID_Test[i]) + '_T1_N3.mnc ' + path_av_t1 + ' --order 1 --noclamp --expfile ' + path_Temp + str(
                    ID_Test[i]) + '_T1_norm --clobber'
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                new_command = 'minccalc -expfile ' + path_Temp + str(ID_Test[i]) + '_T1_norm ' + path_Temp + str(
                    ID_Test[i]) + '_T1_N3.mnc ' + path_Temp + str(ID_Test[i]) + '_T1_VP.mnc '
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_VP.mnc'
                str_main_modality = str_t1_proc

            new_command = 'bestlinreg_s2 ' + str_t1_proc + ' ' + path_av_t1 + ' ' + path_Temp + str(
                ID_Test[i]) + '_T1toTemplate_pp_lin.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'mincresample ' + str_t1_proc + ' -transform ' + path_Temp + str(
                ID_Test[i]) + '_T1toTemplate_pp_lin.xfm' + ' ' + path_Temp + str(
                ID_Test[i]) + '_T1_lin.mnc -like ' + path_av_t1 + ' -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'nlfit_s ' + path_Temp + str(
                ID_Test[i]) + '_T1_lin.mnc ' + path_av_t1 + ' ' + path_Temp + str(
                ID_Test[i]) + '_T1toTemplate_pp_nlin.xfm -level 2 -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'xfmconcat ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm ' + path_Temp + str(
                ID_Test[i]) + '_T1toTemplate_pp_nlin.xfm ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_both.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
