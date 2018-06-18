#Prepared by Arpan Mukherjee
from apt_importers import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help='Input File Name')
parser.add_argument('--rangefile', required=True, help='Range File')
flags = parser.parse_args()



epos = read_file(flags.filename)
ions, rrngs = read_rrng(flags.rangefile)

data = epos.loc[:, ['x', 'y', 'z']].values


lpos = label_ions(epos, rrngs)
dpos = deconvolve(lpos)
volvis(dpos)
