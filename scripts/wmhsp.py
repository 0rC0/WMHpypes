# -*- coding: utf-8 -*-

"""
Created on Wed Jul 15 12:43:32 2017
######################################################################################################################################
Copyright (c)  2018 Mahsa Dadar, Louis Collins

Script for Segmenting White Matter Hyperintensities from T1-w, T2-w, PD, and FLAIR images
This version of the pipeline has the option to perform its own preprocessing and works with .nii images.
 
Input arguments:
WMH_Segmentation_Pipeline.py -c <Classifier (Default: LDA)> -i <Input CSV File> 
 -m <Template Mask File>  -f <Number of Folds in K-fold Cross Validation (Default=10)>
 -o <Output Path> -t <Temp Files Path> -e <Classification Mode> -n <New Data CSV File> -p <Pre-trained Classifiers Path> -d  <Do Preprocessing>
 
CSV File Column Headers: Subjects, XFMs, T1s, T2s, PDs, FLAIRs, WMHs, cls, Masks
Subjects:   Subject ID
T1s:        Path to preprocessed T1 image, coregistered with primary modality (mandatory)
T2s:        Path to preprocessed T2 image, coregistered with primary modality (if exists)
PD:         Path to preprocessed PD image, coregistered with primary modality (if exists)
FLAIR:      Path to preprocessed FLAIR image, coregistered with primary modality (if exists)
XFMs:       Nonlinear transformation from primary modality image to template
Masks:      Brain mask or mask of region of interest
cls:        Tissue classification, where WM=3, GM=2, and CSF=1 (This option has not been used for pre-Training)
WMHs:       White Matter Hyperintensity Labels (For Training, not necessary if using pre-trained classifiers)
 
Preprocessing Options: 
 Y:   Perform Preprocessing 

Classification Mode Options: 
 CV:   Cross Validation (On The Same Dataset) 
 TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)
 PT:   Using Pre-trained Classifiers  
 
Classifier Options:
 NB:   Naive Bayes
 LDA:  Linear Discriminant Analysis
 QDA:  Quadratic Discriminant Analysis
 LR:   Logistic Regression
 KNN:  K Nearest Neighbors 
 RF:   Random Forest 
 SVM:  Support Vector Machines 
 Tree: Decision Tree
 Bagging
 AdaBoost
#####################################################################################################################################
@author: mdadar
"""

#import joblib

def doPreprocessing(path_nlin_mask,path_Temp, ID_Test, WMH_Files_Test , wmh, T1_Files_Test , t1 , T2_Files_Test , t2 , PD_Files_Test , pd , FLAIR_Files_Test , flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair):
    import os

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
    preprocessed_list_address=path_Temp+'Preprocessed.csv'
    print('Preprocessing Images')
    for i in range(0 , len(T1_Files_Test)):
        if (t1 != ''):
            str_File_t1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):
                new_command = 'nii2mnc ' + str_File_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1.mnc'
            else:
                new_command = 'cp ' + str_File_t1 + ' ' + path_Temp + str(ID_Test[i]) + '_T1.mnc'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = '/bestlinreg_s2 ' +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' +  path_av_t1 + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'mincresample ' +  path_nlin_mask + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc -invert_transform -like ' + path_Temp + str(ID_Test[i]) + '_T1.mnc -nearest -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1.mnc'
            str_main_modality = str_t1_proc
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_T1.mnc ' + path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc -beta 0.7 -clobber'
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc'
                str_main_modality = str_t1_proc
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_T1_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_T1_N3.mnc -iter 200 -distance 50 -clobber'
                #-mask '+ path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_N3.mnc'
                str_main_modality = str_t1_proc
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_T1_N3.mnc '  + path_av_t1 + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_T1_norm --clobber'
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                new_command = 'minccalc -expfile ' + path_Temp + str(ID_Test[i]) + '_T1_norm '  + path_Temp + str(ID_Test[i]) + '_T1_N3.mnc ' + path_Temp + str(ID_Test[i]) + '_T1_VP.mnc '
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_t1_proc = path_Temp + str(ID_Test[i]) + '_T1_VP.mnc'
                str_main_modality = str_t1_proc
                
            new_command = 'bestlinreg_s2 ' +  str_t1_proc + ' ' +  path_av_t1 + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'mincresample ' +  str_t1_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T1_lin.mnc -like ' + path_av_t1 + ' -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'nlfit_s ' + path_Temp + str(ID_Test[i]) + '_T1_lin.mnc ' +  path_av_t1 + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_nlin.xfm -level 2 -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'xfmconcat ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_lin.xfm ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_nlin.xfm '+ path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_both.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)

                
        if (t2 != ''):
            str_File_t2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):                
                new_command = 'nii2mnc ' + str_File_t2 + ' ' + path_Temp + str(ID_Test[i]) + '_T2.mnc'      
            else:
                new_command = 'cp ' + str_File_t2 + ' ' + path_Temp + str(ID_Test[i]) + '_T2.mnc'
            os.system(new_command)
            new_command = 'bestlinreg_s2 -lsq6 ' +  path_Temp + str(ID_Test[i]) + '_T2.mnc '  +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' + ' ' +  path_Temp + str(ID_Test[i]) + '_T2toT1.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc -transform ' +  path_Temp + str(ID_Test[i]) + '_T2toT1.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T2_Mask.mnc -invert_transform -like ' + path_Temp + str(ID_Test[i]) + '_T2.mnc -nearest -clobber'
            os.system(new_command)
            str_t2_proc = path_Temp + str(ID_Test[i]) + '_T2.mnc'
            str_main_modality = str_t2_proc
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_T2.mnc ' + path_Temp + str(ID_Test[i]) + '_T2_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_t2_proc = path_Temp + str(ID_Test[i]) + '_T2_NLM.mnc'
                str_main_modality = str_t2_proc
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_T2_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_T2_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_T2_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_t2_proc = path_Temp + str(ID_Test[i]) + '_N3.mnc'
                str_main_modality = str_t2_proc
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_T2_N3.mnc '  + path_av_t2 + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_T2_norm --clobber'
                os.system(new_command)
                new_command = 'minccalc -expfile ' + path_Temp + str(ID_Test[i]) + '_T2_norm '  + path_Temp + str(ID_Test[i]) + '_T2_N3.mnc ' + path_Temp + str(ID_Test[i]) + '_T2_VP.mnc ' 
                os.system(new_command)
                str_t2_proc = path_Temp + str(ID_Test[i]) + '_T2_VP.mnc'
                str_main_modality = str_t2_proc
                         
            
        if (pd != ''):
            str_File_pd = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):                
                new_command = 'nii2mnc ' + str_File_pd + ' ' + path_Temp + str(ID_Test[i]) + '_PD.mnc'    
            else:
                new_command = 'cp ' + str_File_pd + ' ' + path_Temp + str(ID_Test[i]) + '_PD.mnc'    
            os.system(new_command)
            new_command = 'bestlinreg_s2 -lsq6 ' +  path_Temp + str(ID_Test[i]) + '_PD.mnc '  +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' + ' ' +  path_Temp + str(ID_Test[i]) + '_PDtoT1.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc -transform ' +  path_Temp + str(ID_Test[i]) + '_PDtoT1.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_PD_Mask.mnc -invert_transform -like ' + path_Temp + str(ID_Test[i]) + '_PD.mnc -nearest -clobber'
            os.system(new_command)
            str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD.mnc'
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_PD.mnc ' + path_Temp + str(ID_Test[i]) + '_PD_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD_NLM.mnc'
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_PD_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_PD_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_PD_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD_N3.mnc'
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_PD_N3.mnc '  + path_av_pd + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_PD_norm --clobber'
                os.system(new_command)
                new_command = 'minccalc -expfile ' + path_Temp + str(ID_Test[i]) + '_PD_norm '  + path_Temp + str(ID_Test[i]) + '_PD_N3.mnc ' + path_Temp + str(ID_Test[i]) + '_PD_VP.mnc ' 
                os.system(new_command)
                str_pd_proc = path_Temp + str(ID_Test[i]) + '_PD_VP.mnc'
                
        if (flair != ''):
            str_File_flair = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):                
                new_command = 'nii2mnc ' + str_File_flair + ' ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc'
            else:
                new_command = 'cp ' + str_File_flair + ' ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'bestlinreg_s2 -lsq6 ' +  path_Temp + str(ID_Test[i]) + '_FLAIR.mnc '  +  path_Temp + str(ID_Test[i]) + '_T1.mnc ' + ' ' +  path_Temp + str(ID_Test[i]) + '_FLAIRtoT1.xfm'
            print('Executing: {}'.format(new_command))
            os.system(new_command)
            new_command = 'mincresample ' +  path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc -transform ' +  path_Temp + str(ID_Test[i]) + '_FLAIRtoT1.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_FLAIR_Mask.mnc -invert_transform -like ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc -nearest -clobber'
            print('Executing: {}'.format(new_command))
            os.system(new_command)

            str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR.mnc'
            str_main_modality = str_flair_proc
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_FLAIR.mnc ' + path_Temp + str(ID_Test[i]) + '_FLAIR_NLM.mnc -beta 0.7 -clobber'
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR_NLM.mnc'
                str_main_modality = str_flair_proc
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_FLAIR_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc  -iter 200 -distance 50 -clobber'
                # -mask '+ path_Temp + str(ID_Test[i]) + '_FLAIR_Mask.mnc
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc'
                str_main_modality = str_flair_proc
            if (volpolf == 'Y'):
                print('Executing: {}'.format(new_command))
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc '  + path_av_flair + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_FLAIR_norm --clobber'
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                print('Executing: {}'.format(new_command))
                new_command = 'minccalc -expfile ' + path_Temp + str(ID_Test[i]) + '_FLAIR_norm '  + path_Temp + str(ID_Test[i]) + '_FLAIR_N3.mnc ' + path_Temp + str(ID_Test[i]) + '_FLAIR_VP.mnc '
                print('Executing: {}'.format(new_command))
                os.system(new_command)
                str_flair_proc = path_Temp + str(ID_Test[i]) + '_FLAIR_VP.mnc'
                str_main_modality = str_flair_proc
        if (wmh != ''):
            str_File_wmh = str(WMH_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):   
                new_command = 'nii2mnc ' + str_File_wmh + ' ' + path_Temp + str(ID_Test[i]) + '_WMH.mnc' 
            else:
                new_command = 'cp ' + str_File_wmh + ' ' + path_Temp + str(ID_Test[i]) + '_WMH.mnc'
            os.system(new_command)
            str_File_wmh = path_Temp + str(ID_Test[i]) + '_WMH.mnc' 
        
        if ((str_t1_proc != str_main_modality) & (t1 != '')):
            new_command = 'bestlinreg_s2 -lsq6 ' +  str_t1_proc + ' '  +  str_main_modality + ' ' +  path_Temp + str(ID_Test[i]) + '_T1toMain.xfm  -source_mask ' + path_Temp + str(ID_Test[i]) + '_T1_Mask.mnc -clobber'
            os.system(new_command)
            new_command = 'mincresample ' +  str_t1_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T1toMain.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T1_CR.mnc -like ' + str_main_modality + ' -clobber'
            os.system(new_command)
            str_t1_proc =  path_Temp + str(ID_Test[i]) + '_T1_CR.mnc'        
                                   
        if ((str_t2_proc != str_main_modality) & (t2 != '')):
            new_command = 'bestlinreg_s2 -lsq6 ' +  str_t2_proc + ' '  +  str_main_modality + ' ' +  path_Temp + str(ID_Test[i]) + '_T2toMain.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  str_t2_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_T2toMain.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_T2_CR.mnc -like ' + str_main_modality + ' -clobber'
            os.system(new_command)
            str_t1_proc =  path_Temp + str(ID_Test[i]) + '_T2_CR.mnc'
        
        if (pd != ''):    
            new_command = 'bestlinreg_s2 -lsq6 ' +  str_pd_proc + ' '  +  str_main_modality + ' ' +  path_Temp + str(ID_Test[i]) + '_PDtoMain.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  str_pd_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_PDtoMain.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_PD_CR.mnc -like ' + str_main_modality + ' -clobber'
            os.system(new_command)
            str_pd_proc =  path_Temp + str(ID_Test[i]) + '_PD_CR.mnc'
            
            
        if (str_t1_proc != str_main_modality):
            new_command = 'xfminvert ' + path_Temp + str(ID_Test[i]) + '_T1toMain.xfm ' + path_Temp + str(ID_Test[i]) + '_inv_T1toMain.xfm -clobber'
            os.system(new_command)
            new_command = 'xfmconcat ' + path_Temp + str(ID_Test[i]) + '_T1toTemplate_pp_both.xfm ' + path_Temp + str(ID_Test[i]) + '_inv_T1toMain.xfm '+ path_Temp + str(ID_Test[i]) + '_MaintoTemplate.xfm -clobber'
            os.system(new_command)
            new_command = 'mincresample ' +  path_nlin_mask + ' -transform ' + path_Temp + str(ID_Test[i]) + '_MaintoTemplate.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_Mask_nl.mnc -like ' + str_main_modality + ' -invert_transformation -nearest -clobber'
            os.system(new_command) 
            str_Mask = path_Temp + str(ID_Test[i]) + '_Mask_nl.mnc'
            nl_xfm = path_Temp + str(ID_Test[i]) + '_MaintoTemplate.xfm'
            
        print('.'),
        preprocessed_list[0,0]= 'Subjects,T1s,Masks,XFMs'
        preprocessed_list[i+1,0]= str(ID_Test[i]) + ',' + str_t1_proc + ',' + str_Mask + ',' + nl_xfm
        if (t2 != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',T2s'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_t2_proc
        if (pd != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',PDs'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_pd_proc
        if (flair != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',FLAIRs'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_flair_proc
        if (wmh != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',WMHs'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_File_wmh
            
    outfile = open( preprocessed_list_address, 'w' )
    for key, value in sorted( preprocessed_list.items() ):
        outfile.write(  str(value) + '\n' )
    outfile = open( preprocessed_list_address, 'w' )
    for key, value in sorted( preprocessed_list.items() ):
        outfile.write(  str(value) + '\n' )
    return [preprocessed_list_address]
##########################################################################################################################################################
def Calculate_Average(Files_Train , Masks_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K):
    import minc
    import numpy as np
    import os
    str_File=str(Files_Train[0]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
    str_Mask=str(Masks_Train[0]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
    nl_xfm=str(XFM_Files_Train[0]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
    print('Calculating the averages: .'),
    new_command = 'itk_resample --like  ' + path_nlin_mask + ' --transform ' + nl_xfm + ' ' + str_File + ' ' + path_Temp + str(K) + '_tmp_Nonlin_File.mnc --clobber'
    os.system(new_command)
    new_command = 'itk_resample --like  ' + path_nlin_mask + ' --transform ' + nl_xfm + ' ' + str_Mask + ' ' + path_Temp + str(K) + '_tmp_Nonlin_WMH.mnc --label --clobber'
    os.system(new_command)
    manual_segmentation = minc.Image(path_Temp + str(K) + '_tmp_Nonlin_WMH.mnc').data
    image_vol = minc.Image(path_Temp + str(K) + '_tmp_Nonlin_File.mnc').data
    if (np.max(manual_segmentation)!=0):
        manual_segmentation = np.round(manual_segmentation / np.max(manual_segmentation))    
    sp = manual_segmentation
    av_vol = image_vol * (1 - manual_segmentation)

    for i in range(1 , Files_Train.size):
        str_File = str(Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_Mask = str(Masks_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        print('.'),
        new_command = 'itk_resample --like  ' + path_nlin_mask + ' --transform ' + nl_xfm + ' ' + str_File + ' ' + path_Temp + str(K) + '_tmp_Nonlin_File.mnc --clobber'
        os.system(new_command)
        new_command = 'itk_resample --like  ' + path_nlin_mask + ' --transform ' + nl_xfm + ' ' + str_Mask + ' ' + path_Temp + str(K) + '_tmp_Nonlin_WMH.mnc --label --clobber'
        os.system(new_command)
        manual_segmentation = minc.Image(path_Temp + str(K) + '_tmp_Nonlin_WMH.mnc').data
        image_vol = minc.Image(path_Temp + str(K) + '_tmp_Nonlin_File.mnc').data
        if (np.max(manual_segmentation)!=0):
            manual_segmentation = np.round(manual_segmentation / np.max(manual_segmentation))
        sp = sp + manual_segmentation
        av_vol = av_vol + image_vol * (1 - manual_segmentation)

    sp = sp / Files_Train.size    
    av_vol = av_vol / ((1 - sp) * Files_Train.size)
    print(' Done.')
    return [sp , av_vol]
###########################################################################################################################################################################
def Calculate_Tissue_Histogram(Files_Train , Masks_Train , WMH_Files_Train , image_range):
    import minc
    import numpy as np
    
    PDF_WMH = np.zeros(shape = (image_range , 1)).astype(float)
    PDF_Healthy_Tissue = np.zeros(shape = (image_range , 1)).astype(float)
    print('Calculating Histograms of Tissues: .'),
    for i in range(0 , len(Files_Train)):
        print('.'),
        str_File = str(Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_Mask = str(Masks_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_WMH = str(WMH_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        manual_segmentation = minc.Image(str_WMH).data
        if (np.max(manual_segmentation)!=0):
            manual_segmentation = np.round(manual_segmentation / np.max(manual_segmentation))
        image_vol = minc.Image(str_File).data
        brain_mask = minc.Image(str_Mask).data
        image_vol = np.round(image_vol)
        for j in range(1 , image_range):
            PDF_Healthy_Tissue[j] = PDF_Healthy_Tissue[j] + np.sum((image_vol * (1 - manual_segmentation) * brain_mask) == j)
            PDF_WMH[j] = PDF_WMH[j] + np.sum((image_vol * manual_segmentation * brain_mask) == j)
    
    PDF_Healthy_Tissue = PDF_Healthy_Tissue / np.sum(PDF_Healthy_Tissue)
    PDF_WMH = PDF_WMH / np.sum(PDF_WMH)
    print(' Done.')
    return [PDF_Healthy_Tissue , PDF_WMH]
###########################################################################################################################################################################
def load_csv(csv_file):
    import csv
    data = {}
    with open(csv_file , 'r') as f:
        for r in csv.DictReader(f):
            for k in r.keys(): # python2 -> iterkeys():
                k1 = k.lstrip().rstrip()
                try:
                    data[k1].append(r[k])
                except KeyError:
                    data[k1] = [r[k]]
    return data
###########################################################################################################################################################################
def get_Train_Test(Indices_G , K , IDs):
    import numpy as np
    i_train = 0
    i_test = 0
    ID_Train = np.empty(shape = (np.sum(Indices_G != K) , 1) , dtype = list , order = 'C')
    ID_Test = np.empty(shape = (np.sum(Indices_G == K) , 1) , dtype = list , order = 'C')        
    for i in range(0 , len(Indices_G)):
        if (Indices_G[i] != K):
            ID_Train[i_train] = IDs[i]
            i_train = i_train + 1
        if (Indices_G[i] == K):
            ID_Test[i_test] = IDs[i]
            i_test = i_test + 1
    return [ID_Train , ID_Test]
###########################################################################################################################################################################
def get_addressess(TestList):
          
    InputListInfo_Test = load_csv(TestList)
    print('get_adrresses',InputListInfo_Test)
    ID_Test = InputListInfo_Test['Subjects']
    if 'XFMs' in InputListInfo_Test:    
        XFM_Files_Test = InputListInfo_Test['XFMs']
        xfmf = 'exists'
    else:
        xfmf = ''
        XFM_Files_Test = ''
    if 'Masks' in InputListInfo_Test:    
        Mask_Files_Test = InputListInfo_Test['Masks']
        maskf = 'exists'
    else:
        maskf = ''
        Mask_Files_Test = ''
    if 'T1s' in InputListInfo_Test:    
        T1_Files_Test = InputListInfo_Test['T1s']
        print('T1_Files_Test',T1_Files_Test)
        t1 = 'exists'
    else:
        t1 =''
        T1_Files_Test = ''
    if 'T2s' in InputListInfo_Test:    
        T2_Files_Test = InputListInfo_Test['T2s']
        t2 = 'exists'
    else:
        t2 =''
        T2_Files_Test = ''
    if 'PDs' in InputListInfo_Test:    
        PD_Files_Test = InputListInfo_Test['PDs']
        pd = 'exists'
    else:
        pd = ''
        PD_Files_Test = ''
    if 'FLAIRs' in InputListInfo_Test:    
        FLAIR_Files_Test = InputListInfo_Test['FLAIRs']
        flair = 'exists'
    else:
        flair = ''
        FLAIR_Files_Test = ''
    if 'WMHs' in InputListInfo_Test:    
        WMH_Files_Test = InputListInfo_Test['WMHs']
        wmh = 'exists'
    else:
        wmh = ''
        WMH_Files_Test = ''
    if 'cls' in InputListInfo_Test:    
        cls_Files_Test = InputListInfo_Test['cls']
        clsf = 'exists'
    else:
        clsf = ''
        cls_Files_Test = ''
    print('return',[ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, WMH_Files_Test, wmh,cls_Files_Test, clsf])
    return [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, WMH_Files_Test, wmh,cls_Files_Test, clsf]
    
###########################################################################################################################################################################
import sys,getopt
def main(argv):

    import minc
    import numpy as np
    import os
    #from joblib import Parallel
    #import multiprocessing
# Default Values    
    n_folds=10
    image_range = 256
    subject = 0
    Classifier='LDA'    
    InputList=''
    doPreprocessingf='N'
    try:
        opts, args = getopt.getopt(argv,"hc:i:m:o:t:e:n:f:p:d:",["cfile=","ifile=","mfile=","ofile=","tfile=","efile=","nfile=","ffile=","pfile=","dfile="])
    except getopt.GetoptError:
        print('WMH_Segmentation_Pipeline.py -c <Classifier (Default: LDA)> -i <Input CSV File> \n -m <Template Mask File> -f <Number of Folds in K-fold Cross Validation (Default=10)>')
        print(' -o <Output Path> -t <Temp Files Path> -e <Classification Mode> -n <New Data CSV File> -p <Pre-trained Classifiers Path> -d  <Do Preprocessing>\n ')
        print('CSV File Column Headers: Subjects, XFMs, T1s, T2s, PDs, FLAIRs, WMHs, cls, Masks\n')
        print('Preprocessing Options: \n Y:   Perform Preprocessing \n')
        print('Classification Mode Options: \n CV:   Cross Validation (On The Same Dataset) \n TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)\n')
        print(' PT:   Using Pre-trained Classifiers \n'                  )
        print('Classifier Options:\n NB:   Naive Bayes\n LDA:  Linear Discriminant Analysis\n QDA:  Quadratic Discriminant Analysis\n LR:   Logistic Regression')
        print(' KNN:  K Nearest Neighbors \n RF:   Random Forest \n SVM:  Support Vector Machines \n Tree: Decision Tree\n Bagging\n AdaBoost')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('WMH_Segmentation_Pipeline.py -c <Classifier (Default: LDA)> -i <Input CSV File> \n -m <Template Mask File>  -f <Number of Folds in K-fold Cross Validation (Default=10)>')
            print(' -o <Output Path> -t <Temp Files Path> -e <Classification Mode> -n <New Data CSV File> -p <Pre-trained Classifiers Path> -d  <Do Preprocessing>\n '            )
            print('CSV File Column Headers: Subjects, XFMs, T1s, T2s, PDs, FLAIRs, WMHs, cls, Masks\n'            )
            print('Preprocessing Options: \n Y:   Perform Preprocessing \n'            )
            print('Classification Mode Options: \n CV:   Cross Validation (On The Same Dataset) \n TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)')
            print(' PT:   Using Pre-trained Classifiers \n'            )
            print('Classifier Options:\n NB:   Naive Bayes\n LDA:  Linear Discriminant Analysis\n QDA:  Quadratic Discriminant Analysis\n LR:   Logistic Regression')
            print(' KNN:  K Nearest Neighbors \n RF:   Random Forest \n SVM:  Support Vector Machines \n Tree: Decision Tree\n Bagging\n AdaBoost')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            Classifier = arg
            print('Classfr')
        elif opt in ("-i", "--ifile"):
            InputList = arg
        elif opt in ("-m", "--mfile"):
            path_nlin_mask = arg
        elif opt in ("-o", "--ofile"):
            path_output = arg
        elif opt in ("-t", "--tfile"):
            path_Temp = arg+str(np.random.randint(1000000, size=1)).replace("[",'').replace("]",'').replace(" ",'').replace(" ",'')+'_'
        elif opt in ("-e", "--efile"):
            ClassificationMode = arg
            print('ClassMode')
        elif opt in ("-n", "--nfile"):
            TestList = arg
        elif opt in ("-f", "--ffile"):
            n_folds = int(arg)
        elif opt in ("-p", "--pfile"):
            path_trained_classifiers = arg
        elif opt in ("-d", "--dfile"):
            doPreprocessingf = arg            

    print('The Selected Input CSV File is ', InputList)
    print('The Selected Classifier is ', Classifier)
    print('The Classification Mode is ', ClassificationMode)
    print('The Selected Template Mask is ', path_nlin_mask)
    print('The Selected Output Path is ', path_output    )
    print('The Assigned Temp Files Path is ', path_Temp)
    if (doPreprocessingf == 'Y'):
        print('Preprocessing:  Yes')
    #print 'The Pre-trained Classifiers Path is ', path_trained_classifiers
    
    if (Classifier == 'NB'):
        # Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()        
    elif (Classifier == 'LDA'):      
        # Linear Discriminant Analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(solver = "svd" , store_covariance = True)  
    elif (Classifier == 'QDA'):
        # Quadratic Discriminant Analysis
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis(priors = None, reg_param = 0.0 , store_covariances = False , tol = 0.0001)
    elif (Classifier == 'LR'):
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C = 200 , penalty = 'l2', tol = 0.01)
    elif (Classifier == 'KNN'):
        # K Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors = 10)
    elif (Classifier == 'Bagging'):
        # Bagging
        from sklearn.ensemble import BaggingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        clf = BaggingClassifier(KNeighborsClassifier() , max_samples = 0.5 , max_features = 0.5)        
    elif (Classifier == 'AdaBoost'):
        # AdaBoost
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators = 100)        
    elif (Classifier == 'RF'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 100)
    elif (Classifier == 'SVM'):
        # Support Vector Machines
        from sklearn import svm
        clf = svm.LinearSVC()
    elif (Classifier == 'Tree'):
        # Decision Tree
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()   
    else:
        print('The Selected Classifier Was Not Recongnized')
        sys.exit()
    [IDs, XFM_Files, xfmf, Mask_Files, maskf, T1_Files, t1, T2_Files, t2, PD_Files, pd, FLAIR_Files, flair, WMH_Files, wmh,cls_Files, clsf] = get_addressess(InputList)
####################### Preprocessing ####################################################################################################################################

    path_av_t1=path_trained_classifiers+'Av_T1.mnc'
    path_av_t2=path_trained_classifiers+'Av_T2.mnc'
    path_av_pd=path_trained_classifiers+'Av_PD.mnc'
    path_av_flair=path_trained_classifiers+'Av_FLAIR.mnc'      
    ###########################################################################################################################################################################
    if ClassificationMode == 'CV':
        if (doPreprocessingf == 'Y'):
            preprocessed_list_address=doPreprocessing(path_nlin_mask,path_Temp, IDs, WMH_Files , wmh, T1_Files , t1 , T2_Files , t2 , PD_Files , pd , FLAIR_Files, flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [IDs, XFM_Files, xfmf, Mask_Files, maskf, T1_Files, t1, T2_Files, t2, PD_Files, pd, FLAIR_Files, flair, WMH_Files, wmh,cls_Files, clsf] = get_addressess(path_Temp+'Preprocessed.csv')

        if wmh == '':    
            print('No WMH Labels to Train on')
            sys.exit()
        Indices_G = np.random.permutation(len(IDs)) * n_folds / len(IDs)
        Kappa = np.zeros(shape = (len(IDs) , 1))
        ID_Subject = np.empty(shape = (len(IDs),1) , dtype = list, order = 'C')
        #num_cores = multiprocessing.cpu_count()
        #Parallel(n_jobs = num_cores) 
        for K in range(0 , n_folds):
            [ID_Train , ID_Test] = get_Train_Test(Indices_G , K , IDs)    
            [XFM_Files_Train , XFM_Files_Test] = get_Train_Test(Indices_G , K , XFM_Files)    
            [WMH_Files_Train , WMH_Files_Test] = get_Train_Test(Indices_G , K , WMH_Files)    
            [Mask_Files_Train , Mask_Files_Test] = get_Train_Test(Indices_G , K , Mask_Files)    
            
            
            if (clsf != ''):
                n_features=2
                [cls_Files_Train , cls_Files_Test] = get_Train_Test(Indices_G , K , cls_Files)   
            else:
                n_features=1
                
            if (t1 != ''):        
                [T1_Files_Train , T1_Files_Test] = get_Train_Test(Indices_G , K , T1_Files)    
                [spatial_prior , av_T1] = Calculate_Average(T1_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
                path_av_t1 = path_Temp + str(K) + '_av_t1.mnc'
                out = minc.Image(data = av_T1)
                out.save(name = path_av_t1 , imitate = path_nlin_mask)
                [T1_PDF_Healthy_Tissue , T1_PDF_WMH] = Calculate_Tissue_Histogram(T1_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
                n_features = n_features + 5
                
            if (t2 != ''):
                [T2_Files_Train , T2_Files_Test] = get_Train_Test(Indices_G , K , T2_Files)    
                [spatial_prior , av_T2] = Calculate_Average(T2_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
                path_av_t2 = path_Temp + str(K) + '_av_t2.mnc'
                out = minc.Image(data = av_T2)
                out.save(name = path_av_t2 , imitate = path_nlin_mask)
                [T2_PDF_Healthy_Tissue , T2_PDF_WMH] = Calculate_Tissue_Histogram(T2_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
                T2_PDF_WMH[np.argmax(T2_PDF_WMH) : len(T2_PDF_WMH)] = T2_PDF_WMH[np.argmax(T2_PDF_WMH)] # test this
                n_features = n_features + 5
                if (t1 != ''):
                    n_features=n_features+1
                
            if (pd != ''):
                [PD_Files_Train , PD_Files_Test] = get_Train_Test(Indices_G , K , PD_Files)    
                [spatial_prior , av_PD] = Calculate_Average(PD_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
                path_av_pd = path_Temp + str(K) + '_av_pd.mnc'
                out = minc.Image(data = av_PD)
                out.save(name = path_av_pd , imitate = path_nlin_mask)
                [PD_PDF_Healthy_Tissue , PD_PDF_WMH] = Calculate_Tissue_Histogram(PD_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
                PD_PDF_WMH[np.argmax(PD_PDF_WMH):len(PD_PDF_WMH)]=PD_PDF_WMH[np.argmax(PD_PDF_WMH)] # test this
                n_features = n_features + 5
                if (t1 != ''):
                    n_features=n_features+1
                
            if (flair != ''):
                [FLAIR_Files_Train , FLAIR_Files_Test] = get_Train_Test(Indices_G , K , FLAIR_Files)    
                [spatial_prior , av_FLAIR] = Calculate_Average(FLAIR_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
                path_av_flair = path_Temp + str(K) + '_av_flair.mnc'
                out = minc.Image(data = av_FLAIR)
                out.save(name = path_av_flair , imitate = path_nlin_mask)
                [FLAIR_PDF_Healthy_Tissue , FLAIR_PDF_WMH] = Calculate_Tissue_Histogram(FLAIR_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
                FLAIR_PDF_WMH[np.argmax(FLAIR_PDF_WMH) : len(FLAIR_PDF_WMH)] = FLAIR_PDF_WMH[np.argmax(FLAIR_PDF_WMH)] # see if both is better or cdf
                n_features = n_features + 6
                if (t1 != ''):
                    n_features=n_features+1
            
            path_sp = path_Temp + str(K) + '_SP.mnc'
            out = minc.Image(data = spatial_prior)
            out.save(name = path_sp , imitate = path_nlin_mask)
            
            X_All = np.empty(shape = (0 , n_features) , dtype = float , order = 'C')
            Y_All = np.empty(shape = (0 , ) , dtype = float , order = 'C')
    
            for i in range(0 , len(ID_Train)):
                str_Train = str(ID_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print('Extracting The Features: Subject: ID = ' + str_Train)
                
                FT = np.zeros(shape = (image_range , 1)).astype(float)  
                str_Mask = str(Mask_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                Mask = minc.Image(str_Mask).data
                ind_WM = (Mask > 0)
                wmh = str(WMH_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'') 
                WMT = minc.Image(wmh).data
                if len(np.unique(WMT)) > 2:
                    print('Warning: Non-Binary WMH Label - ID: ' + str_Train)
                    WMT = np.round(WMT / np.max(WMT))
                if (clsf != ''):
                    str_CLS = str(cls_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    CLS = minc.Image(str_CLS).data
                    wm = (CLS == 3) # see performance without this feature
                nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                new_command = 'mincresample ' + path_sp + ' -like  ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_sp.mnc -clobber'
                os.system(new_command)
                spatial_prior = minc.Image(path_Temp + str(K) + '_tmp_sp.mnc').data
                
                if (t1 != ''):
                    str_T1 = str(T1_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T1 = minc.Image(str_T1).data
                    new_command = 'mincresample ' + path_av_t1 + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_t1.mnc -clobber'
                    print('Executing {}'.format(new_command))
                    os.system(new_command)
                    av_T1 = minc.Image(path_Temp + str(K) + '_tmp_t1.mnc').data
                    T1n = np.round(T1)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(T1n * Mask == j)
                    T1 = T1 * np.argmax(T1_PDF_Healthy_Tissue) / np.argmax(FT)
                    T1[T1 < 1] = 1
                    T1[T1 > (image_range - 1)] = (image_range - 1)
                    T1_WM_probability = T1_PDF_Healthy_Tissue[np.round(T1[ind_WM]).astype(int)]
                    T1_WMH_probability = T1_PDF_WMH[np.round(T1[ind_WM]).astype(int)]
                    T1_WM_probability[T1[ind_WM] < 1] = 1
                    T1_WMH_probability[T1[ind_WM] < 1] = 0
                    N = len(T1_WMH_probability)
                    X_t1 = np.zeros(shape = (N , 2))
                    X_t1[0 : N , 0] = T1[ind_WM]
                    X_t1[0 : N , 1] = av_T1[ind_WM]
                    X_t1 = np.concatenate((X_t1 , T1_WMH_probability , T1_WM_probability , (T1_WMH_probability + 0.0001) / (T1_WM_probability + 0.0001)) , axis = 1)
    
                if (t2 != ''):                
                    str_T2 = str(T2_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T2 = minc.Image(str_T2).data
                    new_command = 'mincresample ' + path_av_t2 + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_t2.mnc -clobber'
                    os.system(new_command)
                    av_T2 = minc.Image(path_Temp + str(K) + '_tmp_t2.mnc').data
                    T2n = np.round(T2)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(T2n * Mask == j)
                    T2 = T2 * np.argmax(T2_PDF_Healthy_Tissue) / np.argmax(FT)
                    T2[T2 < 1] = 1
                    T2[T2 > (image_range - 1)] = (image_range - 1)
                    T2_WM_probability = T2_PDF_Healthy_Tissue[np.round(T2[ind_WM]).astype(int)]
                    T2_WMH_probability = T2_PDF_WMH[np.round(T2[ind_WM]).astype(int)]
                    T2_WM_probability[T2[ind_WM] < 1] = 1
                    T2_WMH_probability[T2[ind_WM] < 1] = 0
                    N = len(T2_WMH_probability)
                    if (t1 == ''):
                        X_t2 = np.zeros(shape = (N , 2))
                        X_t2[0 : N , 0] = T2[ind_WM]
                        X_t2[0 : N , 1] = av_T2[ind_WM]
                    if (t1 != ''):
                        X_t2 = np.zeros(shape = (N , 3))
                        X_t2[0 : N , 0] = T2[ind_WM]
                        X_t2[0 : N , 1] = av_T2[ind_WM]    
                        X_t2[0 : N , 2] = T2[ind_WM] / T1[ind_WM]
                        
                    X_t2 = np.concatenate((X_t2 , T2_WMH_probability , T2_WM_probability , (T2_WMH_probability + 0.0001) / (T2_WM_probability + 0.0001)) , axis = 1)
                    
                if (pd != ''):
                    str_PD = str(PD_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    PD = minc.Image(str_PD).data
                    new_command = 'mincresample ' + path_av_pd + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform '+ path_Temp + str(K) + '_tmp_pd.mnc -clobber'
                    os.system(new_command)
                    av_PD = minc.Image(path_Temp + str(K) + '_tmp_pd.mnc').data
                    PDn = np.round(PD)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(PDn * Mask == j)
                    PD = PD * np.argmax(PD_PDF_Healthy_Tissue) / np.argmax(FT)
                    PD[PD < 1] = 1
                    PD[PD > (image_range - 1)] = (image_range - 1)
                    PD_WM_probability = PD_PDF_Healthy_Tissue[np.round(PD[ind_WM]).astype(int)]
                    PD_WMH_probability = PD_PDF_WMH[np.round(PD[ind_WM]).astype(int)]
                    PD_WM_probability[PD[ind_WM] < 1] = 1
                    PD_WMH_probability[PD[ind_WM] < 1] = 0
                    N = len(PD_WMH_probability)
                    if (t1 == ''):
                        X_pd = np.zeros(shape = (N , 2))
                        X_pd[0 : N , 0] = PD[ind_WM]
                        X_pd[0 : N , 1] = av_PD[ind_WM]
                    if (t1 != ''):
                        X_pd = np.zeros(shape = (N , 3))
                        X_pd[0 : N , 0] = PD[ind_WM]
                        X_pd[0 : N , 1] = av_PD[ind_WM]                        
                        X_pd[0 : N , 2] = PD[ind_WM] / T1[ind_WM]
                        
                    X_pd = np.concatenate((X_pd , PD_WMH_probability , PD_WM_probability , (PD_WMH_probability + 0.0001) / (PD_WM_probability + 0.0001)) , axis = 1)                
                    
                if (flair != ''):
                    str_FLAIR = str(FLAIR_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    FLAIR = minc.Image(str_FLAIR).data
                    new_command = 'mincresample ' + path_av_flair + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_flair.mnc -clobber'
                    os.system(new_command)
                    av_FLAIR = minc.Image(path_Temp + str(K) + '_tmp_flair.mnc').data
                    FLAIRn = np.round(FLAIR)
                    for j in range(1,image_range):
                        FT[j] = FT[j] + np.sum(FLAIRn * Mask == j)
                    FLAIR = FLAIR * np.argmax(FLAIR_PDF_Healthy_Tissue) / np.argmax(FT)
                    FLAIR[FLAIR < 1] = 1
                    FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                    FLAIR_WM_probability = FLAIR_PDF_Healthy_Tissue[np.round(FLAIR[ind_WM]).astype(int)]
                    FLAIR_WMH_probability = FLAIR_PDF_WMH[np.round(FLAIR[ind_WM]).astype(int)]
                    FLAIR_WM_probability[FLAIR[ind_WM] < 1] = 1
                    FLAIR_WMH_probability[FLAIR[ind_WM] < 1] = 0
                    N = len(FLAIR_WMH_probability)
                    if (t1 == ''):
                        X_flair = np.zeros(shape = (N , 3))
                        X_flair[0 : N , 0] = FLAIR[ind_WM]
                        X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                        X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                    if (t1 != ''):
                        X_flair = np.zeros(shape = (N , 4))
                        X_flair[0 : N , 0] = FLAIR[ind_WM]
                        X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                        X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                        X_flair[0 : N , 3] = FLAIR[ind_WM] / T1[ind_WM]
                        
                    X_flair = np.concatenate((X_flair , FLAIR_WMH_probability , FLAIR_WM_probability , (FLAIR_WMH_probability + 0.0001) / (FLAIR_WM_probability + 0.0001)) , axis = 1)
                
                if (clsf != ''):
                    X = np.zeros(shape = (N , 2))
                    X[0 : N , 1] = wm[ind_WM] * spatial_prior[ind_WM]
                else:
                    X = np.zeros(shape = (N , 1))
                    X[0 : N , 0] = spatial_prior[ind_WM]    
                if (t1 != ''):
                    X = np.concatenate((X , X_t1) , axis = 1)
                if (t2 != ''):
                    X = np.concatenate((X , X_t2) , axis = 1)
                if (pd != ''):
                    X = np.concatenate((X , X_pd) , axis = 1)
                if (flair != ''):
                    X = np.concatenate((X , X_flair) , axis = 1)
                
                X_All = np.concatenate((X_All , X) , axis = 0)
                Y = np.zeros(shape = (N , ))
                Y[0 : N , ] = (WMT[ind_WM])    
                Y_All = np.concatenate((Y_All , Y) , axis = 0)
            
            # Training the Classifier
            clf = clf.fit(X_All , Y_All)
    
            for i in range(0 , len(ID_Test)):
                str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print('Segmenting Volumes: Subject: ID = ' + str_Test)
                wmh = str(WMH_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                WMT = minc.Image(wmh).data
                if (len(np.unique(WMT)) > 2):
                    print('Warning: Non-Binary WMH Label - ID: ' + str_Test)
                    WMT = np.round(WMT / np.max(WMT))
                    
                str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                Mask = minc.Image(str_Mask).data
                ind_WM = (Mask > 0)
                nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                if (clsf != ''):
                    str_CLS = str(cls_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    CLS = minc.Image(str_CLS).data
                    wm = (CLS == 3)
                new_command = 'mincresample '+ path_sp +' -like  ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_sp.mnc -clobber'
                os.system(new_command)
                spatial_prior = minc.Image(path_Temp + str(K) + '_tmp_sp.mnc').data
                
                FT = np.zeros(shape = (image_range , 1)).astype(float)
                if (t1 != ''):
                    str_T1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T1 = minc.Image(str_T1).data
                    new_command = 'mincresample ' + path_av_t1 + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_t1.mnc -clobber'
                    os.system(new_command)
                    av_T1 = minc.Image(path_Temp + str(K) + '_tmp_t1.mnc').data
                    T1n = np.round(T1)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(T1n * Mask == j)
                    T1 = T1 * np.argmax(T1_PDF_Healthy_Tissue) / np.argmax(FT)
                    T1[T1 < 1] = 1
                    T1[T1 > (image_range - 1)] = (image_range - 1)
                    T1_WM_probability = T1_PDF_Healthy_Tissue[np.round(T1[ind_WM]).astype(int)]
                    T1_WMH_probability = T1_PDF_WMH[np.round(T1[ind_WM]).astype(int)]
                    T1_WM_probability[T1[ind_WM] < 1] = 1
                    T1_WMH_probability[T1[ind_WM] < 1] = 0
                    N = len(T1_WMH_probability)
                    X_t1 = np.zeros(shape = (N , 2))
                    X_t1[0 : N , 0] = T1[ind_WM]
                    X_t1[0 : N , 1] = av_T1[ind_WM]
                    X_t1 = np.concatenate((X_t1 , T1_WMH_probability , T1_WM_probability , (T1_WMH_probability + 0.0001) / (T1_WM_probability + 0.0001)) , axis = 1)
    
                if (t2 != ''):                
                    str_T2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    T2 = minc.Image(str_T2).data
                    new_command = 'mincresample ' + path_av_t2 + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_t2.mnc -clobber'
                    os.system(new_command)
                    av_T2 = minc.Image(path_Temp + str(K) + '_tmp_t2.mnc').data
                    T2n = np.round(T2)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(T2n * Mask == j)
                    T2 = T2 * np.argmax(T2_PDF_Healthy_Tissue) / np.argmax(FT)
                    T2[T2 < 1] = 1
                    T2[T2 > (image_range - 1)] = (image_range - 1)
                    T2_WM_probability = T2_PDF_Healthy_Tissue[np.round(T2[ind_WM]).astype(int)]
                    T2_WMH_probability = T2_PDF_WMH[np.round(T2[ind_WM]).astype(int)]
                    T2_WM_probability[T2[ind_WM] < 1] = 1
                    T2_WMH_probability[T2[ind_WM] < 1] = 0
                    N = len(T2_WMH_probability)
                    if (t1 == ''):
                        X_t2 = np.zeros(shape = (N , 2))
                        X_t2[0 : N , 0] = T2[ind_WM]
                        X_t2[0 : N , 1] = av_T2[ind_WM]
                    if (t1 != ''):
                        X_t2 = np.zeros(shape = (N , 3))
                        X_t2[0 : N , 0] = T2[ind_WM]
                        X_t2[0 : N , 1] = av_T2[ind_WM]    
                        X_t2[0 : N , 2] = T2[ind_WM] / T1[ind_WM]
                    X_t2 = np.concatenate((X_t2 , T2_WMH_probability , T2_WM_probability , (T2_WMH_probability + 0.0001) / (T2_WM_probability + 0.0001)) , axis = 1)
                    
                if (pd != ''):                
                    str_PD = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    PD = minc.Image(str_PD).data
                    new_command = 'mincresample ' + path_av_pd + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp+str(K) + '_tmp_pd.mnc -clobber'
                    os.system(new_command)
                    av_PD = minc.Image(path_Temp + str(K) + '_tmp_pd.mnc').data
                    PDn = np.round(PD)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(PDn * Mask == j)
                    PD = PD * np.argmax(PD_PDF_Healthy_Tissue) / np.argmax(FT)
                    PD[PD < 1] = 1
                    PD[PD > (image_range - 1)] = (image_range - 1)
                    PD_WM_probability = PD_PDF_Healthy_Tissue[np.round(PD[ind_WM]).astype(int)]
                    PD_WMH_probability = PD_PDF_WMH[np.round(PD[ind_WM]).astype(int)]
                    PD_WM_probability[PD[ind_WM] < 1] = 1
                    PD_WMH_probability[PD[ind_WM] < 1] = 0
                    N = len(PD_WMH_probability)
                    if (t1 == ''):
                        X_pd = np.zeros(shape = (N , 2))
                        X_pd[0 : N , 0] = PD[ind_WM]
                        X_pd[0 : N , 1] = av_PD[ind_WM]
                    if (t1 != ''):
                        X_pd = np.zeros(shape = (N , 3))
                        X_pd[0 : N , 0] = PD[ind_WM]
                        X_pd[0 : N , 1] = av_PD[ind_WM]                        
                        X_pd[0 : N , 2] = PD[ind_WM] / T1[ind_WM]
                    X_pd = np.concatenate((X_pd , PD_WMH_probability , PD_WM_probability , (PD_WMH_probability + 0.0001) / (PD_WM_probability + 0.0001)) , axis = 1)     
                    
                if (flair != ''):                
                    str_FLAIR = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    FLAIR = minc.Image(str_FLAIR).data
                    new_command = 'mincresample ' + path_av_flair + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + str(K) + '_tmp_flair.mnc -clobber'
                    os.system(new_command)
                    av_FLAIR = minc.Image(path_Temp + str(K) + '_tmp_flair.mnc').data
                    FLAIRn = np.round(FLAIR)
                    for j in range(1 , image_range):
                        FT[j] = FT[j] + np.sum(FLAIRn * Mask == j)
                    FLAIR = FLAIR * np.argmax(FLAIR_PDF_Healthy_Tissue) / np.argmax(FT)
                    FLAIR[FLAIR < 1] = 1
                    FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                    FLAIR_WM_probability = FLAIR_PDF_Healthy_Tissue[np.round(FLAIR[ind_WM]).astype(int)]
                    FLAIR_WMH_probability = FLAIR_PDF_WMH[np.round(FLAIR[ind_WM]).astype(int)]
                    FLAIR_WM_probability[FLAIR[ind_WM] < 1] = 1
                    FLAIR_WMH_probability[FLAIR[ind_WM] < 1] = 0
                    N = len(FLAIR_WMH_probability)
                    if (t1 == ''):
                        X_flair = np.zeros(shape = (N , 3))
                        X_flair[0 : N , 0] = FLAIR[ind_WM]
                        X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                        X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                    if (t1 != ''):
                        X_flair = np.zeros(shape = (N , 4))
                        X_flair[0 : N , 0] = FLAIR[ind_WM]
                        X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                        X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                        X_flair[0 : N , 3] = FLAIR[ind_WM] / T1[ind_WM]
                        
                    X_flair = np.concatenate((X_flair , FLAIR_WMH_probability , FLAIR_WM_probability , (FLAIR_WMH_probability + 0.0001) / (FLAIR_WM_probability + 0.0001)) , axis = 1)
                
                if (clsf != ''):
                    X = np.zeros(shape = (N , 2))
                    X[0 : N , 1] = wm[ind_WM] * spatial_prior[ind_WM]
                else:
                    X = np.zeros(shape = (N , 1))
                    X[0 : N , 0] = spatial_prior[ind_WM]
                if (t1 != ''):
                    X = np.concatenate((X , X_t1) , axis = 1)
                if (t2 != ''):
                    X = np.concatenate((X , X_t2) , axis = 1)
                if (pd != ''):
                    X = np.concatenate((X , X_pd) , axis = 1)
                if (flair != ''):
                    X = np.concatenate((X , X_flair) , axis = 1)
    
                Y = np.zeros(shape = (N , ))
                Y[0 : N] = WMT[ind_WM]
                Binary_Output = clf.predict(X)
                Kappa[subject] = 2 * np.sum(Y * Binary_Output) / (np.sum(Y) + np.sum(Binary_Output))
                ID_Subject[subject] = ID_Test[i]
                if (np.sum(Y) + np.sum(Binary_Output)) == 0:
                    Kappa[subject] = 1
                print(Kappa[subject])
                subject = subject + 1
                    
                WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0,:]) , len(Mask[0 , 0 , :])))
                WMT_auto[ind_WM] = Binary_Output[0 : N]
                out = minc.Image(data = WMT_auto)
                out.save(name = path_output + Classifier + '_' + str_Test + '_WMH.mnc', imitate = str_Mask)
    
        print('Cross Validation Successfully Completed. \nKappa Values:\n')
        print(Kappa)
        print('Indices')
        print(Indices_G)
        print('Mean Kappa: ' + str(np.mean(Kappa)) + ' - STD Kappa: ' + str(np.std(Kappa)))  
###########################################################################################################################################################################    
    if ClassificationMode == 'TT':           # not checked, without flair/t1
        K=0
        Indices_G=np.ones(shape = (len(IDs) , 1))
        ID_Train = IDs
        XFM_Files_Train = XFM_Files    
        WMH_Files_Train = WMH_Files   
        Mask_Files_Train = Mask_Files  
        if (clsf != ''):
            cls_Files_Train = cls_Files
        if (t1 != ''):
            T1_Files_Train = T1_Files
        if (t2 != ''):        
            T2_Files_Train = T2_Files
        if (pd != ''):
            PD_Files_Train = PD_Files
        if (flair != ''):
            FLAIR_Files_Train = FLAIR_Files
        if (doPreprocessingf == 'Y'):
            preprocessed_list_address=doPreprocessing(path_nlin_mask,path_Temp, IDs, WMH_Files , wmh, T1_Files , t1 , T2_Files , t2 , PD_Files , pd , FLAIR_Files, flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [IDs, XFM_Files, xfmf, Mask_Files, maskf, T1_Files, t1, T2_Files, t2, PD_Files, pd, FLAIR_Files, flair, WMH_Files, wmh,cls_Files, clsf] = get_addressess(path_Temp+'Preprocessed.csv')

        [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, WMH_Files_Test, wmh,cls_Files_Test, clsf] = get_addressess(TestList)
        ############## Preprocessing ####################################################################################################################################
        if (doPreprocessingf == 'Y'):
            preprocessed_list_address=doPreprocessing(path_nlin_mask,path_Temp, ID_Test, WMH_Files_Test , wmh, T1_Files_Test , t1 , T2_Files_Test , t2 , PD_Files_Test , pd , FLAIR_Files_Test , flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, WMH_Files_Test, wmh,cls_Files_Test, clsf] = get_addressess(preprocessed_list_address)
        
        #num_cores = multiprocessing.cpu_count()
        #Parallel(n_jobs=num_cores) 
        if (clsf != ''):
            n_features = 2
        else:
            n_features = 1
            
        if (t1 != ''):        
            [T1_Files_Train , tmp] = get_Train_Test(Indices_G , K , T1_Files)    
            [spatial_prior , av_T1] = Calculate_Average(T1_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
            path_av_t1 = path_Temp + '_TT_av_t1.mnc'
            out = minc.Image(data = av_T1)
            out.save(name = path_av_t1 , imitate = path_nlin_mask)
            [T1_PDF_Healthy_Tissue , T1_PDF_WMH] = Calculate_Tissue_Histogram(T1_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
            n_features = n_features + 5
                
        if (t2 != ''):
            [T2_Files_Train , tmp] = get_Train_Test(Indices_G , K , T2_Files)    
            [spatial_prior , av_T2] = Calculate_Average(T2_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
            path_av_t2 = path_Temp + '_TT_av_t2.mnc'
            out = minc.Image(data = av_T2)
            out.save(name = path_av_t2 , imitate = path_nlin_mask)
            [T2_PDF_Healthy_Tissue , T2_PDF_WMH] = Calculate_Tissue_Histogram(T2_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
            T2_PDF_WMH[np.argmax(T2_PDF_WMH) : len(T2_PDF_WMH)] = T2_PDF_WMH[np.argmax(T2_PDF_WMH)] # test this
            n_features = n_features + 5
            if (t1 != ''):
                n_features=n_features+1            
                
        if (pd != ''):
            [PD_Files_Train , tmp] = get_Train_Test(Indices_G , K , PD_Files)    
            [spatial_prior , av_PD] = Calculate_Average(PD_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
            path_av_pd = path_Temp + '_TT_av_pd.mnc'
            out = minc.Image(data = av_PD)
            out.save(name = path_av_pd , imitate = path_nlin_mask)
            [PD_PDF_Healthy_Tissue , PD_PDF_WMH] = Calculate_Tissue_Histogram(PD_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
            #PD_PDF_WMH[np.argmax(PD_PDF_WMH):len(PD_PDF_WMH)]=PD_PDF_WMH[np.argmax(PD_PDF_WMH)] # test this
            n_features = n_features + 5
            if (t1 != ''):
                n_features=n_features+1
                
        if (flair != ''):
            [FLAIR_Files_Train , tmp] = get_Train_Test(Indices_G , K , FLAIR_Files)    
            [spatial_prior , av_FLAIR] = Calculate_Average(FLAIR_Files_Train , WMH_Files_Train , XFM_Files_Train , path_nlin_mask , path_Temp , K)
            path_av_flair = path_Temp + '_TT_av_flair.mnc'
            out = minc.Image(data = av_FLAIR)
            out.save(name = path_av_flair , imitate = path_nlin_mask)
            [FLAIR_PDF_Healthy_Tissue , FLAIR_PDF_WMH] = Calculate_Tissue_Histogram(FLAIR_Files_Train , Mask_Files_Train , WMH_Files_Train , image_range)
            FLAIR_PDF_WMH[np.argmax(FLAIR_PDF_WMH) : len(FLAIR_PDF_WMH)] = FLAIR_PDF_WMH[np.argmax(FLAIR_PDF_WMH)] # see if both is better or cdf
            n_features = n_features + 6
            if (t1 != ''):
                n_features=n_features+1
                
        path_sp = path_Temp + '_TT_SP.mnc'
        out = minc.Image(data = spatial_prior)
        out.save(name = path_sp , imitate = path_nlin_mask)
            
        X_All = np.empty(shape = (0 , n_features ) , dtype = float , order = 'C')
        Y_All = np.empty(shape = (0 , ) , dtype = float , order = 'C')
    
        for i in range(0 , len(ID_Train)):
            str_Train = str(ID_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print('Extracting The Features: Subject: ID = ' + str_Train)
                
            FT = np.zeros(shape = (image_range , 1)).astype(float)   
            str_Mask = str(Mask_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            Mask = minc.Image(str_Mask).data
            ind_WM = (Mask > 0)
            wmh = str(WMH_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            WMT= minc.Image(wmh).data
            if (len(np.unique(WMT)) > 2):
                print('Warning: Non-Binary WMH Label - ID: ' + str_Train)
                WMT = np.round(WMT / np.max(WMT))
            if (clsf != ''):        
                str_CLS = str(cls_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                CLS = minc.Image(str_CLS).data
                wm = (CLS == 3) # see performance without this feature
            nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            new_command = 'mincresample ' + path_sp + ' -like  ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_sp.mnc -clobber'
            os.system(new_command)
            spatial_prior = minc.Image(path_Temp + '_TT_tmp_sp.mnc').data
                
            if (t1 != ''):
                str_T1 = str(T1_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T1 = minc.Image(str_T1).data
                new_command = 'mincresample ' + path_av_t1 + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_t1.mnc -clobber'
                os.system(new_command)
                av_T1 = minc.Image(path_Temp + '_TT_tmp_t1.mnc').data
                T1n = np.round(T1)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(T1n * Mask == j)
                T1 = T1 * np.argmax(T1_PDF_Healthy_Tissue) / np.argmax(FT)
                T1[T1 < 1] = 1
                T1[T1 > (image_range - 1)] = (image_range - 1)
                T1_WM_probability = T1_PDF_Healthy_Tissue[np.round(T1[ind_WM]).astype(int)]
                T1_WMH_probability = T1_PDF_WMH[np.round(T1[ind_WM]).astype(int)]
                T1_WM_probability[T1[ind_WM] < 1] = 1
                T1_WMH_probability[T1[ind_WM] < 1] = 0
                N = len(T1_WMH_probability)
                X_t1 = np.zeros(shape = (N , 2))
                X_t1[0 : N , 0] = T1[ind_WM]
                X_t1[0 : N , 1] = av_T1[ind_WM]
                X_t1 = np.concatenate((X_t1 , T1_WMH_probability , T1_WM_probability , (T1_WMH_probability + 0.0001) / (T1_WM_probability + 0.0001)) , axis = 1)
    
            if (t2 != ''):                
                str_T2 = str(T2_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T2 = minc.Image(str_T2).data
                new_command = 'mincresample ' + path_av_t2 + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_t2.mnc -clobber'
                os.system(new_command)
                av_T2 = minc.Image(path_Temp + '_TT_tmp_t2.mnc').data
                T2n = np.round(T2)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(T2n * Mask == j)
                T2 = T2 * np.argmax(T2_PDF_Healthy_Tissue) / np.argmax(FT)
                T2[T2 < 1] = 1
                T2[T2 > (image_range - 1)] = (image_range - 1)
                T2_WM_probability = T2_PDF_Healthy_Tissue[np.round(T2[ind_WM]).astype(int)]
                T2_WMH_probability = T2_PDF_WMH[np.round(T2[ind_WM]).astype(int)]
                T2_WM_probability[T2[ind_WM] < 1] = 1
                T2_WMH_probability[T2[ind_WM] < 1] = 0
                N = len(T2_WMH_probability)
                if (t1 == ''):
                    X_t2 = np.zeros(shape = (N , 2))
                    X_t2[0 : N , 0] = T2[ind_WM]
                    X_t2[0 : N , 1] = av_T2[ind_WM]
                if (t1 != ''):
                    X_t2 = np.zeros(shape = (N , 3))
                    X_t2[0 : N , 0] = T2[ind_WM]
                    X_t2[0 : N , 1] = av_T2[ind_WM]    
                    X_t2[0 : N , 2] = T2[ind_WM] / T1[ind_WM]
                X_t2 = np.concatenate((X_t2 , T2_WMH_probability , T2_WM_probability , (T2_WMH_probability + 0.0001) / (T2_WM_probability + 0.0001)) , axis = 1)
                    
            if (pd != ''):
                str_PD = str(PD_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                PD = minc.Image(str_PD).data
                new_command = 'mincresample ' + path_av_pd + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_pd.mnc -clobber'
                os.system(new_command)
                av_PD = minc.Image(path_Temp + '_TT_tmp_pd.mnc').data
                PDn = np.round(PD)
                for j in range(1,image_range):
                    FT[j] = FT[j] + np.sum(PDn * Mask == j)
                PD = PD * np.argmax(PD_PDF_Healthy_Tissue) / np.argmax(FT)
                PD[PD < 1] = 1
                PD[PD > (image_range - 1)] = (image_range - 1)
                PD_WM_probability = PD_PDF_Healthy_Tissue[np.round(PD[ind_WM]).astype(int)]
                PD_WMH_probability = PD_PDF_WMH[np.round(PD[ind_WM]).astype(int)]
                PD_WM_probability[PD[ind_WM] < 1] = 1
                PD_WMH_probability[PD[ind_WM] < 1] = 0
                N = len(PD_WMH_probability)
                if (t1 == ''):
                    X_pd = np.zeros(shape = (N , 2))
                    X_pd[0 : N , 0] = PD[ind_WM]
                    X_pd[0 : N , 1] = av_PD[ind_WM]
                if (t1 != ''):
                    X_pd = np.zeros(shape = (N , 3))
                    X_pd[0 : N , 0] = PD[ind_WM]
                    X_pd[0 : N , 1] = av_PD[ind_WM]                        
                    X_pd[0 : N , 2] = PD[ind_WM] / T1[ind_WM]
                X_pd=np.concatenate((X_pd , PD_WMH_probability , PD_WM_probability , (PD_WMH_probability + 0.0001) / (PD_WM_probability + 0.0001)) , axis = 1)                
                    
            if (flair != ''):
                str_FLAIR = str(FLAIR_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                FLAIR = minc.Image(str_FLAIR).data
                new_command = 'mincresample ' + path_av_flair + ' -like ' + wmh + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_flair.mnc -clobber'
                os.system(new_command)
                av_FLAIR = minc.Image(path_Temp + '_TT_tmp_flair.mnc').data
                FLAIRn = np.round(FLAIR)
                for j in range(1,image_range):
                    FT[j] = FT[j] + np.sum(FLAIRn * Mask == j)
                FLAIR = FLAIR * np.argmax(FLAIR_PDF_Healthy_Tissue) / np.argmax(FT)
                FLAIR[FLAIR < 1] = 1
                FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                FLAIR_WM_probability = FLAIR_PDF_Healthy_Tissue[np.round(FLAIR[ind_WM]).astype(int)]
                FLAIR_WMH_probability = FLAIR_PDF_WMH[np.round(FLAIR[ind_WM]).astype(int)]
                FLAIR_WM_probability[FLAIR[ind_WM] < 1] = 1
                FLAIR_WMH_probability[FLAIR[ind_WM] < 1] = 0
                N = len(FLAIR_WMH_probability)
                if (t1 == ''):
                    X_flair = np.zeros(shape = (N , 3))
                    X_flair[0 : N , 0] = FLAIR[ind_WM]
                    X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                    X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                if (t1 != ''):
                    X_flair = np.zeros(shape = (N , 4))
                    X_flair[0 : N , 0] = FLAIR[ind_WM]
                    X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                    X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                    X_flair[0 : N , 3] = FLAIR[ind_WM] / T1[ind_WM]
                X_flair = np.concatenate((X_flair , FLAIR_WMH_probability , FLAIR_WM_probability , (FLAIR_WMH_probability + 0.0001) / (FLAIR_WM_probability + 0.0001)) , axis = 1)
                            
            if (clsf != ''):
                X = np.zeros(shape = (N , 2))
                X[0 : N , 1] = wm[ind_WM] * spatial_prior[ind_WM]
            else:
                X = np.zeros(shape = (N , 1))
                X[0 : N , 0] = spatial_prior[ind_WM]    
            if (t1 != ''):
                X = np.concatenate((X , X_t1) , axis = 1)
            if (t2 != ''):
                X = np.concatenate((X , X_t2) , axis = 1)
            if (pd != ''):
                X = np.concatenate((X , X_pd) , axis = 1)
            if (flair !=''):
                X = np.concatenate((X , X_flair) , axis = 1)
                
            X_All = np.concatenate((X_All , X) , axis = 0)
            Y = np.zeros(shape = (N,))
            Y[0 : N ,] = (WMT[ind_WM])
            Y_All = np.concatenate((Y_All , Y) , axis = 0)
            
        # Training the Classifier
        clf = clf.fit(X_All , Y_All)
        saveFlag=0
        if saveFlag == 1:
            #import joblib
            from sklearn.externals import joblib
            path_trained_classifiers='/data/ipl/ipl12/Mahsa/MICCAI_Challenge/Trained_Classifiers/'
            path_save_classifier=path_trained_classifiers+Classifier+'_CLS'+clsf+'_T1'+t1+'_T2'+t2+'_PD'+pd+'_FLAIR'+flair+'.pkl'    
            joblib.dump(clf,path_save_classifier)
            if (t1 != ''):
                joblib.dump(T1_PDF_Healthy_Tissue,path_trained_classifiers+'T1_HT.pkl')
                joblib.dump(T1_PDF_WMH,path_trained_classifiers+'T1_WMH.pkl')
            if (t2 != ''):
                joblib.dump(T2_PDF_Healthy_Tissue,path_trained_classifiers+'T2_HT.pkl')
                joblib.dump(T2_PDF_WMH,path_trained_classifiers+'T2_WMH.pkl')
            if (pd != ''):
                joblib.dump(PD_PDF_Healthy_Tissue,path_trained_classifiers+'PD_HT.pkl')
                joblib.dump(PD_PDF_WMH,path_trained_classifiers+'PD_WMH.pkl')
            if (flair != ''):
                joblib.dump(FLAIR_PDF_Healthy_Tissue,path_trained_classifiers+'FLAIR_HT.pkl')
                joblib.dump(FLAIR_PDF_WMH,path_trained_classifiers+'FLAIR_WMH.pkl')
            sys.exit()
            
        for i in range(0 , len(ID_Test)):
            str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print('Segmenting Volumes: Subject: ID = ' + str_Test)
            str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            Mask = minc.Image(str_Mask).data
            ind_WM = (Mask > 0)
            nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (clsf != ''):            
                str_cls=str(cls_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                CLS = minc.Image(str_cls).data
                wm = (CLS == 3)
            new_command = 'mincresample ' + path_sp + ' -like  ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_sp.mnc -clobber'
            os.system(new_command)
            spatial_prior = minc.Image(path_Temp + '_TT_tmp_sp.mnc').data
                
            FT = np.zeros(shape = (image_range,1)).astype(float)
            if (t1 != ''):
                str_T1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T1 = minc.Image(str_T1).data
                new_command = 'mincresample ' + path_av_t1 + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_t1.mnc -clobber'
                os.system(new_command)
                av_T1 = minc.Image(path_Temp + '_TT_tmp_t1.mnc').data
                T1n = np.round(T1)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(T1n * Mask == j)
                T1 = T1 * np.argmax(T1_PDF_Healthy_Tissue) / np.argmax(FT)
                T1[T1 < 1] = 1
                T1[T1 > (image_range - 1)] = (image_range - 1)
                T1_WM_probability = T1_PDF_Healthy_Tissue[np.round(T1[ind_WM]).astype(int)]
                T1_WMH_probability = T1_PDF_WMH[np.round(T1[ind_WM]).astype(int)]
                T1_WM_probability[T1[ind_WM] < 1] = 1
                T1_WMH_probability[T1[ind_WM] < 1] = 0
                N = len(T1_WMH_probability)
                X_t1 = np.zeros(shape = (N , 2))
                X_t1[0 : N , 0] = T1[ind_WM]
                X_t1[0 : N , 1] = av_T1[ind_WM]
                X_t1 = np.concatenate((X_t1 , T1_WMH_probability , T1_WM_probability , (T1_WMH_probability + 0.0001) / (T1_WM_probability + 0.0001)) , axis = 1)
                    
            if (t2 != ''):                
                str_T2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T2 = minc.Image(str_T2).data
                new_command = 'mincresample ' + path_av_t2 + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' +path_Temp + '_TT_tmp_t2.mnc -clobber'
                os.system(new_command)
                av_T2 = minc.Image(path_Temp + '_TT_tmp_t2.mnc').data
                T2n = np.round(T2)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(T2n * Mask == j)
                T2 = T2 * np.argmax(T2_PDF_Healthy_Tissue) / np.argmax(FT)
                T2[T2 < 1] = 1
                T2[T2 > (image_range - 1)] = (image_range - 1)
                T2_WM_probability = T2_PDF_Healthy_Tissue[np.round(T2[ind_WM]).astype(int)]
                T2_WMH_probability = T2_PDF_WMH[np.round(T2[ind_WM]).astype(int)]
                T2_WM_probability[T2[ind_WM] < 1] = 1
                T2_WMH_probability[T2[ind_WM] < 1] = 0
                N = len(T2_WMH_probability)
                if (t1 == ''):
                    X_t2 = np.zeros(shape = (N , 2))
                    X_t2[0 : N , 0] = T2[ind_WM]
                    X_t2[0 : N , 1] = av_T2[ind_WM]
                if (t1 != ''):
                    X_t2 = np.zeros(shape = (N , 3))
                    X_t2[0 : N , 0] = T2[ind_WM]
                    X_t2[0 : N , 1] = av_T2[ind_WM]    
                    X_t2[0 : N , 2] = T2[ind_WM] / T1[ind_WM]
                X_t2 = np.concatenate((X_t2 , T2_WMH_probability , T2_WM_probability , (T2_WMH_probability + 0.0001) / (T2_WM_probability + 0.0001)) , axis = 1)
                    
            if (pd != ''):                
                str_PD = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                PD = minc.Image(str_PD).data
                new_command = 'mincresample ' + path_av_pd + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_pd.mnc -clobber'
                os.system(new_command)
                av_PD = minc.Image(path_Temp + '_TT_tmp_pd.mnc').data
                PDn = np.round(PD)
                for j in range(1 , image_range):
                    FT[j] = FT[j]+np.sum(PDn * Mask == j)
                PD = PD * np.argmax(PD_PDF_Healthy_Tissue) / np.argmax(FT)
                PD[PD < 1] = 1
                PD[PD > (image_range - 1)] = (image_range - 1)
                PD_WM_probability = PD_PDF_Healthy_Tissue[np.round(PD[ind_WM]).astype(int)]
                PD_WMH_probability = PD_PDF_WMH[np.round(PD[ind_WM]).astype(int)]
                PD_WM_probability[PD[ind_WM] < 1] = 1
                PD_WMH_probability[PD[ind_WM] < 1] = 0
                N = len(PD_WMH_probability)
                if (t1 == ''):
                    X_pd = np.zeros(shape = (N , 2))
                    X_pd[0 : N , 0] = PD[ind_WM]
                    X_pd[0 : N , 1] = av_PD[ind_WM]
                if (t1 != ''):
                    X_pd = np.zeros(shape = (N , 3))
                    X_pd[0 : N , 0] = PD[ind_WM]
                    X_pd[0 : N , 1] = av_PD[ind_WM]                        
                    X_pd[0 : N , 2] = PD[ind_WM] / T1[ind_WM]
                X_pd = np.concatenate((X_pd , PD_WMH_probability , PD_WM_probability , (PD_WMH_probability + 0.0001) / (PD_WM_probability + 0.0001)) , axis = 1)     
                    
            if (flair != ''):                
                str_FLAIR = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                FLAIR = minc.Image(str_FLAIR).data
                new_command = 'mincresample ' + path_av_flair + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_flair.mnc -clobber'
                os.system(new_command)
                av_FLAIR = minc.Image(path_Temp + '_TT_tmp_flair.mnc').data
                FLAIRn = np.round(FLAIR)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(FLAIRn * Mask == j)
                FLAIR = FLAIR * np.argmax(FLAIR_PDF_Healthy_Tissue) / np.argmax(FT)
                FLAIR[FLAIR < 1] = 1
                FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                FLAIR_WM_probability = FLAIR_PDF_Healthy_Tissue[np.round(FLAIR[ind_WM]).astype(int)]
                FLAIR_WMH_probability = FLAIR_PDF_WMH[np.round(FLAIR[ind_WM]).astype(int)]
                FLAIR_WM_probability[FLAIR[ind_WM] < 1] = 1
                FLAIR_WMH_probability[FLAIR[ind_WM] < 1] = 0
                N = len(FLAIR_WMH_probability)
                if (t1 == ''):
                    X_flair = np.zeros(shape = (N , 3))
                    X_flair[0 : N , 0] = FLAIR[ind_WM]
                    X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                    X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                if (t1 != ''):
                    X_flair = np.zeros(shape = (N , 4))
                    X_flair[0 : N , 0] = FLAIR[ind_WM]
                    X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                    X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                    X_flair[0 : N , 3] = FLAIR[ind_WM] / T1[ind_WM]
                X_flair = np.concatenate((X_flair , FLAIR_WMH_probability , FLAIR_WM_probability , (FLAIR_WMH_probability + 0.0001) / (FLAIR_WM_probability + 0.0001)) , axis = 1)
                            
            if (clsf != ''):
                X = np.zeros(shape = (N,2))
                X[0 : N , 1] = wm[ind_WM] * spatial_prior[ind_WM]
            else:
                X = np.zeros(shape = (N , 1))
                X[0 : N , 0] = spatial_prior[ind_WM]                
            if (t1 != ''):
                X = np.concatenate((X,X_t1) , axis = 1)
            if (t2 != ''):
                X = np.concatenate((X,X_t2) , axis = 1)
            if (pd != ''):
                X = np.concatenate((X,X_pd) , axis = 1)
            if (flair != ''):
                X = np.concatenate((X,X_flair) , axis = 1)
    
            Y = np.zeros(shape = (N , ))
            Binary_Output = clf.predict(X)       
            Prob_Output=clf.predict_proba(X)            
            #### Saving results #########################################################################################################################            
            WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])))
            WMT_auto[ind_WM] = Binary_Output[0 : N]
            out = minc.Image(data = WMT_auto)
            str_WMHo= path_output + Classifier + '_' + str_Test
            out.save(name = str_WMHo + '_WMH.mnc', imitate = str_Mask)
            
            Prob_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])))
            Prob_auto[ind_WM] = Prob_Output[0 : N,1]
            out = minc.Image(data = Prob_auto)
            out.save(name = str_WMHo + '_P.mnc', imitate = str_Mask)
            
            if (t1 != ''):            
                new_command = 'minc_qc.pl ' + str_T1 + ' --mask ' + str_WMHo + '_WMH.mnc ' + str_WMHo + '_WMH.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
            if (t2 != ''): 
                new_command = 'minc_qc.pl ' + str_T2  +' '+ str_WMHo + '_T2.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
            if (pd != ''): 
                new_command = 'minc_qc.pl ' + str_PD +' '+ str_WMHo + '_PD.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
            if (flair != ''): 
                new_command = 'minc_qc.pl ' + str_FLAIR+' '+ str_WMHo + '_FLAIR.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
###########################################################################################################################################################################    
    path_sp=path_trained_classifiers+'SP.mnc'
    path_av_t1=path_trained_classifiers+'Av_T1.mnc'
    path_av_t2=path_trained_classifiers+'Av_T2.mnc'
    path_av_pd=path_trained_classifiers+'Av_PD.mnc'
    path_av_flair=path_trained_classifiers+'Av_FLAIR.mnc'
    if ClassificationMode == 'PT':       
        from sklearn.externals import joblib
        #import joblib
        print('TestList ' + TestList)
        print('get_addressess(TestList) ')
        print(get_addressess(TestList))
        path_saved_classifier=path_trained_classifiers+Classifier+'_CLS'+clsf+'_T1'+t1+'_T2'+t2+'_PD'+pd+'_FLAIR'+flair+'.pkl'
        [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, WMH_Files_Test, wmh,cls_Files_Test, clsf] = get_addressess(TestList)            
############## Preprocessing ####################################################################################################################################
        if (doPreprocessingf == 'Y'):
            preprocessed_list_address=doPreprocessing(path_nlin_mask,path_Temp, ID_Test, WMH_Files_Test , wmh, T1_Files_Test , t1 , T2_Files_Test , t2 , PD_Files_Test , pd , FLAIR_Files_Test , flair ,  path_av_t1 , path_av_t2 , path_av_pd , path_av_flair)
            [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, T1_Files_Test, t1, T2_Files_Test, t2, PD_Files_Test, pd, FLAIR_Files_Test, flair, WMH_Files_Test, wmh,cls_Files_Test, clsf] = get_addressess(path_Temp+'Preprocessed.csv')

########## Loading Trained Classifier ##########################################################################################################################            
        print('path_saved_classifier',path_saved_classifier)
        clf=joblib.load(path_saved_classifier)
        if (t1 != ''):
            T1_PDF_Healthy_Tissue=joblib.load(path_trained_classifiers+'T1_HT.pkl')
            T1_PDF_WMH=joblib.load(path_trained_classifiers+'T1_WMH.pkl')
        if (t2 != ''):
            T2_PDF_Healthy_Tissue=joblib.load(path_trained_classifiers+'T2_HT.pkl')
            T2_PDF_WMH=joblib.load(path_trained_classifiers+'T2_WMH.pkl')
        if (pd != ''):
            PD_PDF_Healthy_Tissue=joblib.load(path_trained_classifiers+'PD_HT.pkl')
            PD_PDF_WMH=joblib.load(path_trained_classifiers+'PD_WMH.pkl')
        if (flair != ''):
            FLAIR_PDF_Healthy_Tissue=joblib.load(path_trained_classifiers+'FLAIR_HT.pkl')
            FLAIR_PDF_WMH=joblib.load(path_trained_classifiers+'FLAIR_WMH.pkl')
        for i in range(0 , len(ID_Test)):
            str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print('Segmenting Volumes: Subject: ID = ' + str_Test)
            print('Mask_Files_Test: ')
            print(Mask_Files_Test)
            str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            print('Mask: ' + str_Mask)
            Mask = minc.Image(str_Mask).data
            ind_WM = (Mask > 0)
            nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (clsf != ''):            
                str_cls=str(cls_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                CLS = minc.Image(str_cls).data
                wm = (CLS == 3)
            new_command = 'mincresample ' + path_sp + ' -like  ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_sp.mnc -clobber'
            os.system(new_command)
            spatial_prior = minc.Image(path_Temp + '_TT_tmp_sp.mnc').data
                
            FT = np.zeros(shape = (image_range,1)).astype(float)
            if (t1 != ''):
                str_T1 = str(T1_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print('str_T1', str_T1)
                T1 = minc.Image(str_T1).data
                new_command = 'mincresample ' + path_av_t1 + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_t1.mnc' # -clobber'
                print('Executing {}'.format(new_command))
                os.system(new_command)
                av_T1 = minc.Image(path_Temp + '_TT_tmp_t1.mnc').data
                T1n = np.round(T1)
                for j in range(1 , image_range):
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
                X_t1 = np.zeros(shape = (N , 2))
                X_t1[0 : N , 0] = T1[ind_WM]
                X_t1[0 : N , 1] = av_T1[ind_WM]
                X_t1 = np.concatenate((X_t1 , T1_WMH_probability , T1_WM_probability , (T1_WMH_probability + 0.0001) / (T1_WM_probability + 0.0001)) , axis = 1)
                    
            if (t2 != ''):                
                str_T2 = str(T2_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                T2 = minc.Image(str_T2).data
                new_command = 'mincresample ' + path_av_t2 + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' +path_Temp + '_TT_tmp_t2.mnc -clobber'
                os.system(new_command)
                av_T2 = minc.Image(path_Temp + '_TT_tmp_t2.mnc').data
                T2n = np.round(T2)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(T2n * Mask == j)
                T2 = T2 * np.argmax(T2_PDF_Healthy_Tissue) / np.argmax(FT)
                T2[T2 < 1] = 1
                T2[T2 > (image_range - 1)] = (image_range - 1)
                T2_WM_probability = T2_PDF_Healthy_Tissue[np.round(T2[ind_WM]).astype(int)]
                T2_WMH_probability = T2_PDF_WMH[np.round(T2[ind_WM]).astype(int)]
                T2_WM_probability[T2[ind_WM] < 1] = 1
                T2_WMH_probability[T2[ind_WM] < 1] = 0
                N = len(T2_WMH_probability)
                if (t1 == ''):
                    X_t2 = np.zeros(shape = (N , 2))
                    X_t2[0 : N , 0] = T2[ind_WM]
                    X_t2[0 : N , 1] = av_T2[ind_WM]
                if (t1 != ''):
                    X_t2 = np.zeros(shape = (N , 3))
                    X_t2[0 : N , 0] = T2[ind_WM]
                    X_t2[0 : N , 1] = av_T2[ind_WM]    
                    X_t2[0 : N , 2] = T2[ind_WM] / T1[ind_WM]
                X_t2 = np.concatenate((X_t2 , T2_WMH_probability , T2_WM_probability , (T2_WMH_probability + 0.0001) / (T2_WM_probability + 0.0001)) , axis = 1)
                    
            if (pd != ''):                
                str_PD = str(PD_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                PD = minc.Image(str_PD).data
                new_command = 'mincresample ' + path_av_pd + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_pd.mnc -clobber'
                os.system(new_command)
                av_PD = minc.Image(path_Temp + '_TT_tmp_pd.mnc').data
                PDn = np.round(PD)
                for j in range(1 , image_range):
                    FT[j] = FT[j]+np.sum(PDn * Mask == j)
                PD = PD * np.argmax(PD_PDF_Healthy_Tissue) / np.argmax(FT)
                PD[PD < 1] = 1
                PD[PD > (image_range - 1)] = (image_range - 1)
                PD_WM_probability = PD_PDF_Healthy_Tissue[np.round(PD[ind_WM]).astype(int)]
                PD_WMH_probability = PD_PDF_WMH[np.round(PD[ind_WM]).astype(int)]
                PD_WM_probability[PD[ind_WM] < 1] = 1
                PD_WMH_probability[PD[ind_WM] < 1] = 0
                N = len(PD_WMH_probability)
                if (t1 == ''):
                    X_pd = np.zeros(shape = (N , 2))
                    X_pd[0 : N , 0] = PD[ind_WM]
                    X_pd[0 : N , 1] = av_PD[ind_WM]
                if (t1 != ''):
                    X_pd = np.zeros(shape = (N , 3))
                    X_pd[0 : N , 0] = PD[ind_WM]
                    X_pd[0 : N , 1] = av_PD[ind_WM]                        
                    X_pd[0 : N , 2] = PD[ind_WM] / T1[ind_WM]
                X_pd = np.concatenate((X_pd , PD_WMH_probability , PD_WM_probability , (PD_WMH_probability + 0.0001) / (PD_WM_probability + 0.0001)) , axis = 1)     
                    
            if (flair != ''):                
                str_FLAIR = str(FLAIR_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                FLAIR = minc.Image(str_FLAIR).data
                new_command = 'mincresample ' + path_av_flair + ' -like ' + str_Mask + ' -transform ' + nl_xfm + ' -invert_transform ' + path_Temp + '_TT_tmp_flair.mnc -clobber'
                os.system(new_command)
                av_FLAIR = minc.Image(path_Temp + '_TT_tmp_flair.mnc').data
                FLAIRn = np.round(FLAIR)
                for j in range(1 , image_range):
                    FT[j] = FT[j] + np.sum(FLAIRn * Mask == j)
                FLAIR = FLAIR * np.argmax(FLAIR_PDF_Healthy_Tissue) / np.argmax(FT)
                FLAIR[FLAIR < 1] = 1
                FLAIR[FLAIR > (image_range - 1)] = (image_range - 1)
                FLAIR_WM_probability = FLAIR_PDF_Healthy_Tissue[np.round(FLAIR[ind_WM]).astype(int)]
                FLAIR_WMH_probability = FLAIR_PDF_WMH[np.round(FLAIR[ind_WM]).astype(int)]
                FLAIR_WM_probability[FLAIR[ind_WM] < 1] = 1
                FLAIR_WMH_probability[FLAIR[ind_WM] < 1] = 0
                N = len(FLAIR_WMH_probability)
                if (t1 == ''):
                    X_flair = np.zeros(shape = (N , 3))
                    X_flair[0 : N , 0] = FLAIR[ind_WM]
                    X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                    X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                if (t1 != ''):
                    X_flair = np.zeros(shape = (N , 4))
                    X_flair[0 : N , 0] = FLAIR[ind_WM]
                    X_flair[0 : N , 1] = av_FLAIR[ind_WM]
                    X_flair[0 : N , 2] = spatial_prior[ind_WM] * FLAIR[ind_WM]
                    X_flair[0 : N , 3] = FLAIR[ind_WM] / T1[ind_WM]
                X_flair = np.concatenate((X_flair , FLAIR_WMH_probability , FLAIR_WM_probability , (FLAIR_WMH_probability + 0.0001) / (FLAIR_WM_probability + 0.0001)) , axis = 1)
                            
            if (clsf != ''):
                X = np.zeros(shape = (N,2))
                X[0 : N , 1] = wm[ind_WM] * spatial_prior[ind_WM]
            else:
                X = np.zeros(shape = (N , 1))
                X[0 : N , 0] = spatial_prior[ind_WM]                
            if (t1 != ''):
                X = np.concatenate((X,X_t1) , axis = 1)
            if (t2 != ''):
                X = np.concatenate((X,X_t2) , axis = 1)
            if (pd != ''):
                X = np.concatenate((X,X_pd) , axis = 1)
            if (flair != ''):
                X = np.concatenate((X,X_flair) , axis = 1)
    
            Y = np.zeros(shape = (N , ))
            Binary_Output = clf.predict(X)       
            Prob_Output=clf.predict_proba(X)            
            #### Saving results #########################################################################################################################            
            WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])))
            WMT_auto[ind_WM] = Binary_Output[0 : N]
            out = minc.Image(data = WMT_auto)
            str_WMHo= path_output + Classifier + '_' + str_Test
            out.save(name = str_WMHo + '_WMH.mnc', imitate = str_Mask)
            
            Prob_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])))
            Prob_auto[ind_WM] = Prob_Output[0 : N,1]
            out = minc.Image(data = Prob_auto)
            out.save(name = str_WMHo + '_P.mnc', imitate = str_Mask)
            
            new_command= 'mnc2nii ' + str_WMHo + '_WMH.mnc ' + str_WMHo + '_WMH.nii'
            os.system(new_command)
            
            if (t1 != ''):            
                new_command = 'minc_qc.pl ' + str_T1 + ' --mask ' + str_WMHo + '_WMH.mnc ' + str_WMHo + '_WMH.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
            if (t2 != ''): 
                new_command = 'minc_qc.pl ' + str_T2  +' '+ str_WMHo + '_T2.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
            if (pd != ''): 
                new_command = 'minc_qc.pl ' + str_PD +' '+ str_WMHo + '_PD.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
            if (flair != ''): 
                new_command = 'minc_qc.pl ' + str_FLAIR+' '+ str_WMHo + '_FLAIR.jpg --big --clobber  --image-range 0 200 --mask-range 0 1'
                os.system(new_command)
    #os.system('rm ' + path_Temp + '*')
    print('Segmentation Successfully Completed. ')
if __name__ == "__main__":
   main(sys.argv[1:])   

