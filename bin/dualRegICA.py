import argparse
import os

from meshica import dual_regression as drg
from niio import loaded, write

parser = argparse.ArgumentParser()

parser.add_argument('-subjects', '--subject_list', help='List of subjects to process.',
    required=True, type=str)
parser.add_argument('-dir', '--data_dir', help='Directory where resting state data exists.',
    required=True, type=str)
parser.add_argument('-g', '--group_components', help='Group ICA components.',
    required=True, type=str)
parser.add_argument('-e', '--extension', help='Resting state file extension.',
    required=True, type=str)
parser.add_argument('-od', '--out_dir', help='Output directory.',
    required=True, type=str)
parser.add_argument('-o', '--out_base', help='Output file name for group ICA components.',
    required=True, type=str)
parser.add_argument('-hemi', '--hemisphere', help='Hemisphere to process.',
    required=False, type=str, choices=['L','R'], default='L')
parser.add_argument('-a', '--alpha', help='Bayesian CI alpha value.', 
    required=False, type=float, default=0.05)

args = parser.parse_args()

if not os.path.isfile(args.group_components):
    raise('Group ICA file does not exist.')
else:
    groupICA = loaded.load(args.group_components)

with open(args.subject_list,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

hemimap = {'L': 'CortexLeft',
            'R': 'CortexRight'}

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

dual = drg.DualRegression(hdr_alpha=args.alpha)

for s in subjects:

    temp_file = ''.join([args.data_dir, s, args.extension])

    if os.path.isfile(temp_file):
        temp_file = [temp_file]

        dual.fit(temp_file, groupICA)
        spatial = dual.spatial_components[0]

        out_components = ''.join([args.out_dir, s, args.out_base])
        write.save(spatial, out_components, hemimap[args.hemisphere])