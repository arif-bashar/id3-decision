#!/usr/bin/env python3

import sys
import numpy as np

def main():
    # args
    fileName = sys.argv[1]

    data = np.loadtxt(fileName)
    # Special case of just one data sample will fail
    # without this check!
    if len(data.shape) < 2:
        data = np.array([data])

    # Sort all columns - just retain sorted indices
    # NOT the sorted data to prevent need to resort
    # later on...
    indices = np.argsort(data,axis=0)
    
    # Proceed for each column
    for x in range(data.shape[1]):
        print("Sorting along column number:",x)
        # Go through all data in sorted order
        for y in indices[:,x]:
            # Get one training example in sorted order
            print(data[y,:])
            
if __name__ == "__main__":
    main()
