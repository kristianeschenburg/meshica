import argparse
import os

from meshica import canica as cICA
from niio import write

parser = argparse.ArgumentParser()

parser.add_argument('-subjects', '--subject_list', help='List of subjects to process.',
    required=True, type=str)
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
ica = cICA.CanICA(low_pass=0.1, t_r=0.720)
ica.fit(resting)

hemimap = {'L': 'CortexLeft',
            'R': 'CortexRight'}

print('Saving gICA components...')
write.save(ica.components_, args.output, hemimap[args.hemisphere])