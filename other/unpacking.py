from dataclasses import replace
import gzip
import numpy as np
import pickle

keyFile = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = r'C:\\Users\\Genki\\oithomes\\c\\python\\ZeroTuku_1\\other\MNIST'

for file in keyFile.values():
    gzFile = dataset_dir + '/' + file
    rawFile = dataset_dir + '/' + file.replace(".gz","")
    print("gz_FILE:",file)

    with gzip.open(gzFile,'rb') as fp:
        body = fp.read()
        with open(rawFile,'wb') as w:
            w.write(body)

print("Script End.")