import numpy as np
from analysis import iter_imglist, merge_timestamp, get_vid, get_video_info
import argparse

def get_args():
    parser = argparse.ArgumentParser(\
            description = "Motion stabilization for calcium videos",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input",
            help="input image files.")
    parser.add_argument("-r", "--roi",
            help="roi")
    parser.add_argument("-d", "--debug",
            help = "show debug window",
            action="store_false")
    parser.add_argument("-m", "--max-frame",
            help = "index of the last frame",
            type = int)
    parser.add_argument("-o", "--output",
            help = "output track file, pickled",
            )
    parser.add_argument("-s", "--start-frame",
            help = "start frame number",
            type = int)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    debug = False
    if args.max_frame and args.start_frame:
        frame_shape,vid = get_vid([args.input], [args.max_frame], [args.start_frame])
    else:
        frame_shape,vid = get_vid([args.input])
    print(vid.shape)
    input('pause')
    vid = vid.T.reshape([vid.shape[-1]]+list(frame_shape))
    print(vid.shape)

    if args.output:
        with open(args.output, "wb") as outf:
            np.savez_compressed(outf, vid)
    else:
    	with open(f.split(".")[0].lower(), "wb") as outf:
    		np.savez_compressed(outf, vid)
