import sys, os, importlib
import argparse
import glob
#sys.path.append(os.path.abspath('../'))
from wmhpypes.interfaces import ibbmTum
from wmhpypes.workflows import ibbmTum_wf
from nipype.pipeline.engine import Workflow, Node
from nipype import DataGrabber, DataSink, IdentityInterface, MapNode, JoinNode
from nipype.interfaces.io import BIDSDataGrabber, DataFinder


def make_workflow():

    flairs = [os.path.abspath(i) for i in glob.glob(args.flair)]
    weights = [os.path.abspath(i) for i in glob.glob(args.weights)]
    weights_source = Node(interface=IdentityInterface(fields=['weights']), name='weights_source')
    weights_source.inputs.weights = weights

    data_source = Node(IdentityInterface(fields=['flairs']), name='data_source')
    data_source.iterables = ('flairs', flairs)

    sink = Node(interface=DataSink(), name = 'sink')
    sink.inputs.base_directory = wmh_dir
    sink.inputs.substitutions = [('_flairs_',''),
                                 ('_FLAIR.nii.gz/', '/'),]
    sink.inputs.regexp_substitutions = [('\.\..*\.\.', ''),]

    test_wf = ibbmTum_wf.get_test_wf(row_st=192,
                                     cols_st=192,
                                     thres_mask=10)

    wmh = Workflow(name='wmh', base_dir=wf_temp)

    wmh.connect(weights_source, 'weights', test_wf, 'inputspec.weights')
    wmh.connect(data_source, 'flairs', test_wf, 'inputspec.flair')
    wmh.connect(test_wf, 'outputspec.wmh_mask', sink, '@pred')

    return wmh

def make_dirs():
    global wf_temp, wmh_dir
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    wf_temp = os.path.abspath(os.path.join(args.outdir, 'wmh_temp'))
    wmh_dir = os.path.abspath(os.path.join(args.outdir, 'wmh'))
    if not os.path.isdir(wf_temp):
        os.makedirs(wf_temp)
    if not os.path.isdir(wmh_dir):
        os.makedirs(wmh_dir)


def main():
    print(os.getcwd())
    make_dirs()
    wmh = make_workflow()
    #wmh.write_graph(graph2use='colored')
    #Image('./wf_work_dir/wmh/graph.png', width=200)

    wmh.run() #Single thread
    #plugin_args = {'n_procs': cores}
    #wmh.run(plugin='MultiProc', plugin_args=plugin_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f','--flair', type=str, required=True)
    # parser.add_argument('-m','--mprage', type=str, required=False) # ToDo MPRage
    parser.add_argument('-w','--weights', type=str, required=True)
    parser.add_argument('-o','--outdir', type=str, required=True)
    parser.add_argument('-i','--indir', type=str, required=False, default='')
    args = parser.parse_args()
    main()
