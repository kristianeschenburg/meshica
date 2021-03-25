import argparse
import os

from meshica import migp
from niio import loaded
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser()

parser.add_argument('-files', 
                    '--file-list', 
                    help='List of resting-state files to aggregate.',
                    required=True, 
                    type=str)
parser.add_argument('-c', 
                    '--number-components', 
                    help='Number of ICA components to compute.',
                    required=False, 
                    type=int, 
                    default=20)
parser.add_argument('-lp', 
                    '--low-pass', 
                    help='Low pass filter frequency.',
                    required=False, 
                    type=float, 
                    default=None)
parser.add_argument('-tr', 
                    '--rep-time', 
                    help='Repetition time (TR) in seconds.',
                    required=False, 
                    type=float, 
                    default=0.720)
parser.add_argument('-e', 
                    '--eigens', 
                    help='Number of principcal components to iteratively keep.',
                    required=False, 
                    type=int, 
                    default=3600)
parser.add_argument('-n', 
                    '--number-subjects', 
                    help='Number of subjects to initialize components with.',
                    required=False, 
                    type=int, 
                    default=4)
parser.add_argument('-o', 
                    '--output', 
                    help='Output file name for group ICA components.',
                    required=True, 
                    type=str)
parser.add_argument('-m', 
                    '--mask', 
                    help='Inclusion mask for vertices.', 
                    required=False, 
                    type=str,
                    default=None)
parser.add_argument('-s',
                    '--size',
                    help='Downsample the number of files.',
                    required=False,
                    type=int,
                    default=None)

args = parser.parse_args()

with open(args.file_list, 'r') as f:
    files = f.read().split()
np.random.shuffle(files)

if args.size:
    files = files[:args.size]

if args.mask:
    mask = loaded.load(args.mask)

print('Fitting MIGP with {:} components...'.format(args.number_components))
M = migp.MIGP(n_components=args.number_components, 
              low_pass=args.low_pass,
              m_eigen=args.eigens,
              s_init=args.number_subjects,
              t_r=args.rep_time,
              mask=mask)

M.fit(files)
components = M.components_

if args.mask:
    C = np.zeros((mask.shape[0], components.shape[1]))
    C[np.where(mask),:] = components)
    components = {'components': C}
else:
    components = {'components': components}

print('Saving gICA components...')
sio.savemat(file_name=args.output, mdict=components)
