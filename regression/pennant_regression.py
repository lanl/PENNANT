#!/usr/bin/python
"""
Created on Tue May 31 10:13:51 2016

@author: jgraham
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

EPSILON = 1.0e-11
TOLERANCE = 1.5e-7

def read_xy_file(filename):
    try:
      with open(filename, 'r') as f:
        file_data = f.readlines()
    except IOError:
      file_data = []

    field = []
    values = []
    count = -1
    for line in file_data:
        data = line.split()
        if data[0] == '#':
            field.append(data[1])
            count += 1
            values.append([])
        else:
            values[count].append(float(data[1]))

    return field,np.array(values)

def compare_values(base, regress):
    max_error = 0
    for i in range(len(base)):
        if (np.abs(base[i]) > EPSILON):
            error = np.abs((base[i] - regress[i])/base[i])
        else:
            error = np.abs(base[i] - regress[i])
        if error > max_error:
            max_error = error
    return max_error


parser = argparse.ArgumentParser()
parser.add_argument('base_file', help='baseline .xy.std file')
parser.add_argument('regress_file', help='regression output .xy file')
parser.add_argument('--plot', '-p', dest='plot', action='store_const',
                    const=1, default=0, help='plot comparison')
args = parser.parse_args()

base_fields, base_data = read_xy_file(args.base_file)
regress_fields, regress_data = read_xy_file(args.regress_file)

max_error = 0.0
if len(base_fields) != len(regress_fields):
    print 'Number of fields do not match!'
    max_error = TOLERANCE + 1.0
else:
    for i in range(len(base_fields)):
        if args.plot:
            plt.figure()
            plt.ylabel(base_fields[i])
            plt.plot(base_data[i],'--',label='base')
            plt.plot(regress_data[i],':',label='regress')
            plt.legend(loc='best')
        error = compare_values(base_data[i], regress_data[i])
        print base_fields[i], error
        if error > max_error:
            max_error = error

if args.plot:
    plt.show()
        
if max_error > TOLERANCE:
    print '\x1b[31mFAIL\x1b[0m'
    sys.exit(1)
else:
    print '\x1b[32mPASS\x1b[0m'
    sys.exit(0)
