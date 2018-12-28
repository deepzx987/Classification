import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import csv
import operator

filename = '100.csv'
f = open(filename, 'rb')

reader = csv.reader(f, delimiter=',')
MLII_index = 1
V1_index = 2
MLII = []
V1 = []

next(reader)
MLII_index = 1
V1_index = 2
MLII = []
V1 = []
for row in reader:
    MLII.append((int(row[MLII_index])))
    V1.append((int(row[V1_index])))
f.close()

import time

filename = '100annotations.txt'
f = open(filename, 'rb')
next(f)
annotations = []
for line in f:
    annotations.append(line)
f.close

# CHECK
# use your algorithm also to scheck use SNR wali algo
# # median_filter1D
# baseline = medfilt(MLII, 71)
# baseline = medfilt(baseline, 215)

# # Remove Baseline
# for i in range(0, len(MLII)):
#     MLII[i] = MLII[i] - baseline[i]

for a in annotations:
    aS = a.split()
    pos = int(aS[1])
    originalPos = int(aS[1])
    classAnttd = aS[2]
    if pos > 20 and pos < (len(MLII) - 20):
        index, value = max(enumerate(MLII[pos - 20: pos + 20]), key=operator.itemgetter(1))
        print index, value
        time.sleep(20)
        pos = (pos - 20) + index
        if pos != originalPos:
            print pos, originalPos

for a in annotations:
    aS = a.split()
    pos = int(aS[1])
    if pos > 20 and pos < (len(MLII) - 20):
        print 'POS:'
        print pos

        print 'MLII:'
        print MLII[pos - 20: pos + 20]

        e = enumerate(MLII[pos - 20: pos + 20])
        print list(e)

        m1 = max(enumerate(MLII[pos - 20: pos + 20]))
        print m1

        m2 = max(enumerate(MLII[pos - 20: pos + 20]), key=operator.itemgetter(1))
        print m2

        index, value = max(enumerate(MLII[pos - 20: pos + 20]), key=operator.itemgetter(1))
        print index, value

        break

# change hue hai peaks by one

# EMD me ye ni krna padega to time bachega

# wapis peak detection time leta hai


# Extract the R-peaks from annotations
for a in annotations:
    aS = a.split()

    pos = int(aS[1])
    originalPos = int(aS[1])
    classAnttd = aS[2]
    if pos > size_RR_max and pos < (len(MLII) - size_RR_max):
        index, value = max(enumerate(MLII[pos - size_RR_max: pos + size_RR_max]), key=operator.itemgetter(1))
        pos = (pos - size_RR_max) + index

    peak_type = 0
    # pos = pos-1

    if classAnttd in MITBIH_classes:
        if (pos > winL and pos < (len(MLII) - winR)):
            beat[r].append((MLII[pos - winL: pos + winR], V1[pos - winL: pos + winR]))
            for i in range(0, len(AAMI_classes)):
                if classAnttd in AAMI_classes[i]:
                    class_AAMI = i
                    break  # exit loop
            # convert class
            class_ID[r].append(class_AAMI)

            valid_R[r] = np.append(valid_R[r], 1)
        else:
            valid_R[r] = np.append(valid_R[r], 0)
    else:
        valid_R[r] = np.append(valid_R[r], 0)

    R_poses[r] = np.append(R_poses[r], pos)
    Original_R_poses[r] = np.append(Original_R_poses[r], originalPos)

# R_poses[r] = R_poses[r][(valid_R[r] == 1)]
# Original_R_poses[r] = Original_R_poses[r][(valid_R[r] == 1)]


import pickle, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import csv
import operator

for b in my_db.beat[0]:
    print len(b[1])
    break

s = 0
for i in range(len(my_db.beat)):
    a = len(my_db.valid_R[i]) - len(my_db.beat[i])
    s = s + a
    print len(my_db.beat[i]), 'Valid Beats out of ', len(
        my_db.valid_R[i]), ' for MLII and V1 for file ', i + 1, ' Invalid = ', a
print 'Invalid total = ', s

for i in range(len(my_db.raw_signal)):
    print len(my_db.raw_signal[i][0]), 'length of MLII and ', len(
        my_db.raw_signal[i][1]), 'length of V1 for file ', i + 1

len(my_db.class_ID)

len(my_db.filename)

len(my_db.orig_R_pos)

len(my_db.R_pos)

my_db.valid_R[i]

# features bana rahe hai
# labels hai class_id aur patient_num_beats hai beat of my_db

features.shape

features[0]

len(my_db.class_ID)

# Feature selection yet to do..


# Start modelling if you want

import gc

features_labels_name = 'w_90_90_DS1_rm_bsline_maxRR_u-lbp_MLII.p'

print("Loading pickle: " + features_labels_name + "...")
f = open(features_labels_name, 'rb')
# disable garbage collector
gc.disable()  # this improve the required loading time!
features, labels, patient_num_beats = pickle.load(f)
gc.enable()
f.close()

print features.shape

print labels.shape

print patient_num_beats.shape

len(my_db.beat)


def compute_Uniform_LBP(signal, neigh=8):
    hist_u_lbp = np.zeros(59, dtype=float)

    avg_win_size = 2
    # NOTE: Reduce sampling by half
    # signal_avg = scipy.signal.resample(signal, len(signal) / avg_win_size)

    for i in range(neigh / 2, len(signal) - neigh / 2):
        pattern = np.zeros(neigh)
        ind = 0
        for n in range(-neigh / 2, 0) + range(1, neigh / 2 + 1):
            if signal[i] > signal[i + n]:
                pattern[ind] = 1
            ind += 1
        # Convert pattern to id-int 0-255 (for neigh == 8)
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        # Convert id to uniform LBP id 0-57 (uniform LBP)  58: (non uniform LBP)
        if pattern_id in uniform_pattern_list:
            pattern_uniform_id = int(np.argwhere(uniform_pattern_list == pattern_id))
        else:
            pattern_uniform_id = 58  # Non uniform patterns use

        hist_u_lbp[pattern_uniform_id] += 1.0

    return hist_u_lbp


neigh = 8
for n in range(-neigh / 2, 0) + range(1, neigh / 2 + 1):
    print n

leads_flag = [1, 0]

num_leads = np.sum(leads_flag)

mit_pickle_name = 'python_mit_rm_bsline_wL_90_wR_90_DS1.p'

if os.path.isfile(mit_pickle_name):
    f = open(mit_pickle_name, 'rb')
    my_db = pickle.load(f)
    f.close()

uniform_pattern_list = np.array(
    [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127,
     128,
     129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
     251, 252, 253, 254, 255])

print("u-lbp ...")
features = np.array([], dtype=float)
f_lbp = np.empty((0, 59 * num_leads))

for p in range(len(my_db.beat)):
    for beat in my_db.beat[p]:
        f_lbp_lead = np.empty([])
        for s in range(2):
            if leads_flag[s] == 1:
                if f_lbp_lead.size == 1:

                    f_lbp_lead = compute_Uniform_LBP(beat[s], 8)
                else:
                    f_lbp_lead = np.hstack((f_lbp_lead, compute_Uniform_LBP(beat[s], 8)))
        f_lbp = np.vstack((f_lbp, f_lbp_lead))

features = np.column_stack((features, f_lbp)) if features.size else f_lbp
print(features.shape)

p = 0
for beat in my_db.beat[p]:
    print len(beat[0])
    print len(beat[1])
    print

