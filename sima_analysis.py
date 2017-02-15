import sima
import cv2
import numpy as np
from analysis import get_vid
from matplotlib.pyplot import plot, show
import argparse
import os
import scipy.ndimage

def get_args():
    parser = argparse.ArgumentParser(\
            description = "Analysis of calcium imaging data using PCA and ICA.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", 
                          help="file to save PCA and ICA components, in npz format")
    parser.add_argument("input",
                          help="input image files. A list of tiff / npy files, one frame each, alphanumeric ordered",
                          nargs="*")
    #parser.add_argument("-m", "--max-frames",
    #                      help="maximum number of frames to process",
    #                      type=int,
    #                      default=[],
    #                      nargs = "*")
    #parser.add_argument("-s", "--start-frames",
    #                      help="the frame number to start the analysis",
    #                      type=int,
    #                      default = [],
    #                      nargs = "*")
    parser.add_argument("-n", "--components",
                          help="number of components for ICA",
                          default = 100,
                          type=int)
    parser.add_argument("-T", "--trim",
                          help="trim black pixels, a two-integer or four-interger array indicating number of pixels left/right and number of pixels top/bottom",
                          type=int,
                          nargs=4)
    parser.add_argument("--shift-coords", help="pickled array, with xy shift coords for motion correction", nargs="*", default=[])
    parser.add_argument("--illum-corr-size", help="window size for illumination correction size, recommend 15", type=int)
    parser.add_argument("--rebuild-sources", help="source files used to rebuild time course", nargs="*")
    parser.add_argument("--memmap",
                          help="file name to store video array")
    parser.add_argument("--no-overwrite", help="skip if result file is already present", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for i in range(len(args.input)):
    	frame_shape, vid = get_vid([args.input[i]])
    	data = {str(i): vid.T.reshape([vid.shape[-1], 1, frame_shape[0], frame_shape[1], 1])}

    #input('continue to sima?')
    #SIMA
    # motion correction

    sequences = [sima.Sequence.create('ndarray', data[str(i)]) for i in range(len(args.input))]
    dataset_path = args.output
    correction_approach = sima.motion.PlaneTranslation2D(max_displacement=[30,30])
    if os.path.exists(dataset_path):
    	while True:
    		input_ = input("Dataset path already exists. Overwrite? (y/n) ")
    		if input_ == 'n':
    			exit()
    		elif input_ == 'y':
    			shutil.rmtree(dataset_path)
    			break
    print('Running motion correction')
    dataset = correction_approach.correct(sequences, dataset_path, channel_names=['gcamp'], trim_criterion=0.95)
    #written for only one sequence
    for sequences in dataset:
    	seq = np.ravel(sequences).reshape(sequences.shape[0],sequences.shape[2],sequences.shape[3])
    max_img = np.nanmax(seq,axis=0)
    max_img -=np.min(max_img)
    m = np.max(max_img)
    max_img = np.uint8(max_img/m*255)
    scipy.misc.imsave('max.png', max_img)


    input('continue to segmentation?')


    # segmentation 
    segmentation_approach = sima.segment.STICA(channel='gcamp', components=args.components, verbose=False)
    print('Running autosegmentation')
    rois = dataset.segment(segmentation_approach, 'stica_ROIs')
    #check in ROI body?

    input('continue?')

    # extract the signal
    print('extracting signal') # Export the extracted signals to a CSV file.
    dataset.extract(signal_channel='gcamp', label='gcamp_signals')

    print("Exporting GCaMP time series.")
    dataset.export_signals('dataset_signal.csv', channel='gcamp',signals_label='gcamp_signals')
    input('continue to displaying signal?')

    print("Displaying example calcium trace.")# plot the signal from an ROI object, with a different color for each cycle
    raw_signals = dataset.signals('gcamp')['gcamp_signals']['raw']
    for sequence in range(len(args.input)):  # plot data from the first 3 cycles
	    plot(raw_signals[sequence][3])  # plot the data from ROI #3
