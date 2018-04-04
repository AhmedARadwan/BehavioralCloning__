import csv
# import cv2
# import numpy as np
# from sklearn.utils import shuffle
# import random
# from matplotlib import pyplot as plt

images = []
images_paths = []

measurements = []


files = []

files.append('data')



for f in files:
    lines = []
    with open(f+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    print(float(line[6]))