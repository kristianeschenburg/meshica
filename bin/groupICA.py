import argparse
import os

from meshica import canica as cICA
from niio import write

parser = argparse.ArgumentParser()

parser.add_argument('-subjects', '--subject_list', help='List of subjects to process.',
    required=True, type=str)
parser.add_argument('-nc', '--n_components', help='Number of ICA components to compute.',
    required=False, type=int, default=20)
parser.add_argument('-lp', '--low_pass', help='Low pass filter frequency.',
    required=False, type=float, default=0.1)
parser.add_argument('-tr', '--rep_time', help='Repetition time.',
    required=False, type=float, default=0.720)
parser.add_argument('-dir', '--data_dir', help='Directory where resting state data exists.',
    required=True, type=str)
parser.add_argument('-e', '--extension', help='Resting state file extension.',
    required=True, type=str)
parser.add_argument('-o', '--output', help='Output file name for group ICA components.',
    required=True, type=str)
parser.add_argument('-hemi', '--hemisphere', help='Hemisphere to process.',
    required=False, type=str, choices=['L','R'], default='L')

args = parser.parse_args()

with open(args.subject_list,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

baseDir = os.path.dirname(args.output) + '/'

if not os.path.isdir(baseDir):
    os.mkdir(baseDir)

resting = []
for s in subjects:
    temp_file = ''.join([args.data_dir, s, args.extension])
    
    if os.path.isfile(temp_file):
        resting.append(temp_file)

print('Fitting gICA components...')
ica = cICA.CanICA(n_components=args.n_components, low_pass=args.low_pass, t_r=args.rep_time)
ica.fit(resting)

hemimap = {'L': 'CortexLeft',
            'R': 'CortexRight'}

print('Saving gICA components...')
write.save(ica.components_, args.output, hemimap[args.hemisphere])
