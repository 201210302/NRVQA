import argparse
import os
import sys
import pdb

import cv2
from joblib import load

from brisque import brisque
from niqe import niqe
from piqe import piqe

def main():

    parser = argparse.ArgumentParser('Test an image')
    parser.add_argument(
        '--mode', choices=['brisque', 'niqe', 'piqe'], help='iqa algorithoms,brisque or niqe or piqe')
    parser.add_argument('--path', required=True, help='image path')
    args = parser.parse_args()
    
    mode = args.mode
    

    for img_name in sorted(os.listdir(args.path)):
        img_path = os.path.join(args.path,img_name)
        #print(img_path)

        if not os.path.exists(img_path):
            continue

        #img read
        
        im = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)

        if im is None:
            print("please input correct image path!")
            sys.exit(0)
        if mode == "piqe":
            score, _, _, _ = piqe(im)
        elif mode == "niqe":
            score = niqe(im)
        elif mode == "brisque":
            feature = brisque(im)
            feature = feature.reshape(1, -1)
            clf = load('svr_brisque.joblib')
            score = clf.predict(feature)[0]
        print("{}-----{} score:{}".format(img_name, mode, score))

if __name__ == '__main__':
    main()





