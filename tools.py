# -*- coding: utf-8 -*-
"""
Useful functions. Usually taken from other modules.
"""
import numpy as np

def read_rdb(rdb_file, sepchar='\t', skiprows = 0, headrows = 2):
    """
    Loads data from an rdb file.
    """
    #read file
    f = open(rdb_file,'r')
    lines = f.readlines()
    f.close()

    # read header rows
    header = lines[skiprows].split()
    fmt_all = dict((header[i],i) for i in range(len(header)))
    
    data = {}
    for line in lines[skiprows + headrows:]:
        if line.startswith('#'):
            continue

        elems = line.split(sepchar)

        for i, fmt in enumerate(fmt_all.keys()):

            try:
                elem = elems[fmt_all[fmt]]
            except IndexError:
                print('Problem in this line:')
                print(line)
                print(line.split(sepchar))
                print('Element {0}, at postion {1} '
                      'was not reached.'.format(fmt, i))
                continue

            try:
                elem = float(elem)
            except ValueError:
                #print(fmt, elem)
                pass

            if fmt in data:
                data[fmt].append(elem)
            else:
                data[fmt] = [elem,]

    for dd in data:
        data[dd] = np.array(data[dd])

    return data
