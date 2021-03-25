import argparse, os

from meshica import dual_regression as dr
import numpy as np
from niio import loaded, write
import scipy.io as sio

def main(args):

    components = loaded.load(args.components)
    rest = loaded.load(args.rest)
    
    if args.mask:
        mask = loaded.load(args.mask)
        components = components[np.where(mask)]
        rest = rest[np.where(mask),:]
        rest = rest.squeeze()

    # instantiate dual regressor
    Regressor = dr.Regressor(standardize=args.standardize, 
                             hdr_alpha=args.alpha, 
                             tr=args.rep_time,
                             s_filter=args.filter)
    
    # fit spatial and temporal regression components
    Regressor.fit(rest, components)
    
    temporal = {'temporal': Regressor.temporal_}
    spatial = {'spatial': Regressor.spatial_}

    if args.mask:
        temp = np.zeros((mask.shape[0], spatial['spatial'].shape[1]))
        temp[np.where(mask),:] = spatial['spatial']
        spatial['spatial'] = temp
    
    sio.savemat(file_name='.'.join([args.output, 'Temporal.mat']), mdict=temporal)
    sio.savemat(file_name='.'.join([args.output, 'Spatial.mat']), mdict=spatial)
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Compute single-subject group-ICA maps using dual regression.')

    parser.add_argument('-r', 
                        '--rest', 
                        help='Resting state file for subject.',
                        required=True,
                        type=str)

    parser.add_argument('-c', 
                        '--components', 
                        help='Group ICA components.', 
                        required=True, 
                        type=str)
    
    parser.add_argument('-m', 
                        '--mask', 
                        help='Inclusion mask.', 
                        required=False, 
                        default=None, 
                        type=str)

    parser.add_argument('-o', 
                        '--output', 
                        help='Output base name for spatial and temporal components.', 
                        required=True, 
                        type=str)

    parser.add_argument('-a', 
                        '--alpha', 
                        help='Bayesian confidence interval alpha value.', 
                        required=False, 
                        type=float, 
                        default=None)
    
    parser.add_argument('-tr', 
                        '--rep-time', 
                        help='Repetition time of rs-fmri bold signal.', 
                        required=False, 
                        type=float, 
                        default=0.720)
    
    parser.add_argument('--standardize',
                        help='Temporal standardization of features.',
                        action='store_true',
                        required=False)
    
    parser.add_argument('--filter',
                        help='Apply spectral filtering.',
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    main(args)