import sys

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')


import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
# Wei Yang 2015-08-19
# Source
#   Read LevelDB/LMDB
#   ==================
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
#   Plot image
#   ==================
#       http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
#   Creating LMDB in python
#   ==================
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/

lmdb_file = "./examples/mnist/mnist_test_lmdb"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

f = open("pic.txt",'w')

for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)
    im = data.astype(np.uint8)
    im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    print >>f, "label " + str(label)
    print >>f, im


f.close()
