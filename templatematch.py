# Copyright (c) 2016 Allison Sliter
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from glob import glob
import pickle
from PIL import Image
from heapq import nsmallest
from operator import itemgetter
from collections import defaultdict
from functools import partial
from time import time
from pandas import DataFrame

nums = pickle.load(open("num_temps.pkl", 'rb'))
label_instructions = pickle.load(open("label_with_instructs.pkl", 'rb'))

def argmin_2d(arr):
    if arr.ndim == 1:
        return np.min(arr)
    cell = np.argmin(arr)
    _, n = arr.shape
    return cell/n, cell%n, np.min(arr) 

def match(img, temp):
    x, y = img.shape
    a, b = temp.shape
    assert x == a
    n = max(y,b)
    diff = np.random.rand(x,n)
    diff[:x, :y] = img
    diff[:a, :b] = diff[:a, :b] - temp
    plt.imshow(diff)
    return np.sum(np.abs(diff))

def locate(img, temp, restrict=[]):
    x, y = img.shape
    a, b = temp.shape
    timg = threshold(img)
    ttemp = threshold(temp)
    if not restrict:
        xs, ys = (0,0)
        xe, ye = (x-a, y-b)
    else:
        xs, xe, ys, ye = restrict
    difmat = np.ones((xe-xs, ye-ys))
    for i in xrange(xs, xe):
        for j in xrange(ys, ye):
            difmat[i-xs,j-ys] = np.sum(np.abs(timg[i:i+a, j:j+b] - ttemp))
    return difmat
            
def blankline(img, vert, single):
    m, n = img.shape
    if vert:
        if single:
            return np.zeros(m)
        else:
            return np.zeros((m, 2))
    else:
        if single:
            return np.zeros(n)
        else:
            return np.zeros((2, n))
    

def trim_to_single_white(img):
    copy = img.copy()
    while not np.allclose(copy[:, 0], blankline(copy, True, True)):
        assert copy.shape[1] > 2
        copy = copy[:, 1:]
    while not np.allclose(copy[:, -1],blankline(copy, True, True)):
        assert copy.shape[1] > 2
        copy = copy[:, 0:-1]

    while not np.allclose(copy[0],blankline(copy, False, True)):
        assert copy.shape[0] > 2
        copy = copy[1:]
    while not np.allclose(copy[-1], blankline(copy, False, True)):
        assert copy.shape[0] > 2
        copy = copy[:-1]
    while np.allclose(copy[0:2, :], blankline(copy, False, False)):
        assert copy.shape[0] > 2
        copy = copy[1:]
    while np.allclose(copy[-2:, :], blankline(copy, False, False)):
        assert copy.shape[0] > 2
        copy = copy[:-1]
    while np.allclose(copy[:, :2], blankline(copy, True, False)):
        assert copy.shape[1] > 2
        copy = copy[:, 1:]
    while np.allclose(copy[:, -2:], blankline(copy, True, False)):
        assert copy.shape[1] >2
        copy = copy[:, :-1]
    return copy


def partition_char(img):
    m, n = img.shape
    copy = img.copy()
    blh = np.zeros(n)
    blv = np.zeros(m)
    chars = list()
    start = 0
    end = n
    inchar = False
    for i in xrange(n):
        if np.allclose(copy[:, i], blv):
            if inchar:
                chars.append(copy[:, start:i])
                inchar = False
                start = i
            else:
                start = i
        else:
            if inchar:
                continue
            else:
                start = i
                inchar = True
    return chars

def extract_fields(img, templates):
    hits = dict()
    for k, v in templates.iteritems():
        template, region, offset = v
        match = locate(img, template, restrict=region)
        a,b,c,d = offset
        x,y, _ = argmin_2d(match)
        x += region[0]
        y += region[2]
        hits[k] = img[x+a:x+b, y+c:y+d]
    return hits

def resize(img, new_height):
    m, n = img.shape
    new_width = new_height*n/m
    pilimage = Image.fromarray(img)
    pilimage = pilimage.resize((new_width, new_height), Image.ANTIALIAS)
    arr = np.array(pilimage.getdata())
    t = arr.reshape(new_height, new_width)
    return threshold(t)

def threshold(img):
    copy = img.copy()
    copy[img > np.mean(img)*1.1] = 1
    copy[img < np.mean(img)*1.1] = 0
    return copy


def best_match(img, nums):
    matches = dict()
    for k, v in nums.iteritems():
        score = match(img, v)
        matches[k] = score
    return nsmallest(1, matches.items(), key=itemgetter(1))[0][0]
    

imagefiles = glob("test_graphs/*")
data = defaultdict(partial(defaultdict, str))
# for i in xrange(len(imagefiles)):
#     name = imagefiles[i][12:]
#     print name,
#     img = mpimg.imread(imagefiles[i])
#     try:
#         img = img[:, :, 3]
#     except:
#         print "Wrong format on", imagefiles[i]
#         continue
#     if img.shape[0] < 1000:
#         img = resize(img, img.shape[0]*2)
#     fields = extract_fields(img, label_instructions)
#     for k,v in fields.iteritems():
#         try:
#             chars = partition_char(trim_to_single_white(v))
#         except:
#             print k, "Partition_Fail",
#             continue
#         cell = ""
#         for i in xrange(len(chars)):
#             chars[i] = resize(chars[i], 26)
#             cell+=best_match(chars[i], nums)[0][0]
#         print k, cell,
#     print
#
# df = DataFrame.from_dict(data).T
# df.to_pickle("results_df.pkl")