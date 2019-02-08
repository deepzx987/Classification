import os
import gc
import pickle
from scipy.signal import lfilter, iirfilter, resample
import pandas as pd
# import time

from features import *


def mmf(signal, alpha=0.2):
    return (1 - alpha) * np.median(signal) + alpha * np.mean(signal)

def mean_median_filter(signal, window=300, alpha=0.6):
    # Always odd  window
    if window % 2 == 0:
        window = window - 1

    # Appended signal
    new_signal = []
    for i in range(int((window - 1) / 2)):
        new_signal.append(signal[0])
    for i in range(len(signal)):
        new_signal.append(signal[i])
    for i in range(int((window - 1) / 2)):
        new_signal.append(signal[-1])

    # Windowing
    mmfoutput = []
    for i in range(len(signal)):
        mmfoutput.append(mmf(new_signal[i:i + window], alpha))

    return mmfoutput


def notch_filter(data, freq_to_remove, sample_freq=360.0):
    fs = sample_freq
    nyq = fs / 2.0
    low = freq_to_remove - 1.0
    high = freq_to_remove + 10.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(5, [low, high], btype='bandstop')
    filtered_data = lfilter(b, a, data)
    return filtered_data


def create_features_labels_name(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path,
                                reduced_DS, leads_flag):
    """

    :param DS:
    :param winL:
    :param winR:
    :param do_preprocess:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param compute_morph:
    :param db_path:
    :param reduced_DS:
    :param leads_flag:
    :return:
    """
    features_labels_name = db_path + '/features/'

    if not os.path.exists(features_labels_name):
        os.makedirs(features_labels_name)

    features_labels_name = features_labels_name + 'w_' + str(winL) + '_' + str(winR) + '_' + DS

    if do_preprocess:
        features_labels_name += '_rm_bsline'

    if maxRR:
        features_labels_name += '_maxRR'

    if use_RR:
        features_labels_name += '_RR'

    if norm_RR:
        features_labels_name += '_norm_RR'

    for descp in compute_morph:
        features_labels_name += '_' + descp

    if reduced_DS:
        features_labels_name += '_reduced'

    if leads_flag[0] == 1:
        features_labels_name += '_MLII'

    if leads_flag[1] == 1:
        features_labels_name += '_V1'

    features_labels_name += '.p'

    return features_labels_name


def load_signal(DS, winL, winR, do_preprocess):
    class_ID = [[] for i in range(len(DS))]
    beat = [[] for i in range(len(DS))]  # record, beat, lead
    R_poses = [np.array([]) for i in range(len(DS))]
    Original_R_poses = [np.array([]) for i in range(len(DS))]
    valid_R = [np.array([]) for i in range(len(DS))]
    my_db = mit_db()

    size_RR_max = 20

    pathDB = os.getcwd()
    print (pathDB)
    DB_name = 'data'

    # Read files: signal (.csv )  annotations (.txt)
    fRecords = list()
    fAnnotations = list()

    lst = os.listdir(pathDB + "/" + DB_name + "/csv")
    lst.sort()
    for filename in lst:
        if filename.endswith(".csv"):
            if int(filename[0:3]) in DS:
                fRecords.append(filename)
        elif filename.endswith(".txt"):
            if int(filename[0:3]) in DS:
                fAnnotations.append(filename)

    MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'P', '/', 'f', 'u']
    AAMI_classes = [['N', 'L', 'R'], ['A', 'a', 'J', 'S', 'e', 'j'], ['V', 'E'], ['F'], ['P', '/', 'f', 'u']]

    RAW_signals = []

    # for r, a in zip(fRecords, fAnnotations):
    for r in range(0, len(fRecords)):
        # TODO parallelize the cleaning process
        # pool = multiprocessing.Pool(4)
        # out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
        # We will work only on MLII data as other lead change with the patient
        print ('Processing signal ' + str(r + 1) + ' / ' + str(len(fRecords)) + '...')
        filename = pathDB + "/" + DB_name + "/csv/" + fRecords[r]
        print (filename)
        df = pd.read_csv(filename)
        MLII = df['\'MLII\''].values
        if fRecords[r][:3] == '114' or fRecords[r][:3] == '123' or fRecords[r][:3] == '100':
            V1 = df['\'V5\''].values
        elif fRecords[r][:3] == '117' or fRecords[r][:3] == '103':
            V1 = df['\'V2\''].values
        elif fRecords[r][:3] == '124':
            V1 = df['\'V4\''].values
        else:
            V1 = df['\'V1\''].values

        RAW_signals.append((MLII, V1))  # NOTE a copy must be created in order to preserve the original signal

        # 1. Cleaning the signal
        if do_preprocess:
            MLII = MLII - np.mean(MLII)
            # Remove power_line_interference
            MLII = notch_filter(data=MLII, freq_to_remove=50.0)
            baseline = mean_median_filter(MLII, 300, 0.6)
            baseline = mean_median_filter(baseline, 600, 0.6)
            baseline = np.array(baseline)
            MLII = MLII - baseline

            V1 = V1 - np.mean(V1)
            # Remove power_line_interference
            V1 = notch_filter(data=V1, freq_to_remove=50.0)
            baseline = mean_median_filter(V1, 300, 0.6)
            baseline = mean_median_filter(baseline, 600, 0.6)
            baseline = np.array(baseline)
            V1 = V1 - baseline

        # 2. Read annotations
        filename = pathDB + "/" + DB_name + "/csv/" + fAnnotations[r]
        print (filename)
        data = pd.read_csv(filename, delimiter="\t")

        for i in range(len(data)):
            a = data.values[i][0]
            aa = a.split()
            pos = int(aa[1])
            originalpos = int(aa[1])
            classAnttd = aa[2]
            if size_RR_max < pos < (len(MLII) - size_RR_max):
                # value = max(MLII[pos - size_RR_max: pos + size_RR_max])
                index = np.argmax(MLII[pos - size_RR_max: pos + size_RR_max])
                pos = pos - size_RR_max + index

            if classAnttd in MITBIH_classes:
                if winL < pos < (len(MLII) - winR):
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
            Original_R_poses[r] = np.append(Original_R_poses[r], originalpos)

    # Set the data into a bigger structure that keep all the records!
    my_db.filename = fRecords
    my_db.raw_signal = RAW_signals
    my_db.beat = beat  # record, beat, lead
    my_db.class_ID = class_ID
    my_db.valid_R = valid_R
    my_db.R_pos = R_poses
    my_db.orig_R_pos = Original_R_poses

    return my_db


def load_mit_db(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag):
    # global RR
    print ('Runing train_SVM.py!')

    features_labels_name = create_features_labels_name(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR,
                                                       compute_morph, db_path, reduced_DS, leads_flag)
    print (features_labels_name)

    if os.path.isfile(features_labels_name):
        print ('Features already present')
        print ('Loading pickle: ' + features_labels_name + '...')
        f = open(features_labels_name, 'rb')
        # disable garbage collector
        gc.disable()  # this improve the required loading time!
        features, labels, patient_num_beats = pickle.load(f)
        gc.enable()
        f.close()
    else:
        print ('Generating Features')
        print ("Loading MIT BIH arr (" + DS + ") ...")
        # 102 and 104 do not have the MLII lead data
        # ML-II
        if not reduced_DS:
            DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                   223, 230]
            DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                   233, 234]

        # ML-II + V1
        else:
            DS1 = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
            DS2 = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

        mit_pickle_name = db_path + '/python_mit'

        if leads_flag[0] == 1:
            mit_pickle_name += '_MLII'

        if leads_flag[1] == 1:
            mit_pickle_name += '_V1'

        if reduced_DS:
            mit_pickle_name = mit_pickle_name + '_reduced_'

        if do_preprocess:
            mit_pickle_name = mit_pickle_name + '_rm_bsline'

        mit_pickle_name = mit_pickle_name + '_wL_' + str(winL) + '_wR_' + str(winR) + '_' + DS + '.p'

        print (mit_pickle_name)

        # If the data with that configuration has been already computed Load pickle
        if os.path.isfile(mit_pickle_name):
            print ('Loading Pickle' + mit_pickle_name + '...')
            f = open(mit_pickle_name, 'rb')
            # disable garbage collector
            gc.disable()  # this improve the required loading time!
            my_db = pickle.load(f)
            gc.enable()
            f.close()
        else:  # Load data and compute de RR features
            if DS == 'DS1':
                my_db = load_signal(DS1, winL, winR, do_preprocess)
            else:
                my_db = load_signal(DS2, winL, winR, do_preprocess)

            print("Saving signal processed data ...")
            f = open(mit_pickle_name, 'wb')
            pickle.dump(my_db, f, 2)
            f.close()

        features = np.array([], dtype=float)
        # labels = np.array([], dtype=np.int32)

        # This array contains the number of beats for each patient (for cross_val)
        patient_num_beats = np.array([], dtype=np.int32)
        for p in range(len(my_db.beat)):
            patient_num_beats = np.append(patient_num_beats, len(my_db.beat[p]))

        # Compute RR features
        if use_RR or norm_RR:
            if DS == 'DS1':
                RR = [RR_intervals() for i in range(len(DS1))]
            else:
                RR = [RR_intervals() for i in range(len(DS2))]

            print("Computing RR intervals ...")

            for p in range(len(my_db.beat)):
                if maxRR:
                    RR[p] = compute_rr_intervals(my_db.R_pos[p])
                else:
                    RR[p] = compute_rr_intervals(my_db.orig_R_pos[p])

                # choose all the beats which are valid
                RR[p].pre_R = RR[p].pre_R[(my_db.valid_R[p] == 1)]
                RR[p].post_R = RR[p].post_R[(my_db.valid_R[p] == 1)]
                RR[p].local_R = RR[p].local_R[(my_db.valid_R[p] == 1)]
                RR[p].global_R = RR[p].global_R[(my_db.valid_R[p] == 1)]

        if use_RR:
            f_RR = np.empty((0, 4))
            for p in range(len(RR)):
                row = np.column_stack((RR[p].pre_R, RR[p].post_R, RR[p].local_R, RR[p].global_R))
                f_RR = np.vstack((f_RR, row))

            if features.size:
                features = np.column_stack((features, f_RR))
            else:
                features = f_RR

        if norm_RR:
            f_RR_norm = np.empty((0, 4))
            for p in range(len(RR)):
                # Compute avg values!
                avg_pre_R = np.average(RR[p].pre_R)
                avg_post_R = np.average(RR[p].post_R)
                avg_local_R = np.average(RR[p].local_R)
                avg_global_R = np.average(RR[p].global_R)

                row = np.column_stack((RR[p].pre_R / avg_pre_R, RR[p].post_R / avg_post_R, RR[p].local_R / avg_local_R, RR[p].global_R / avg_global_R))
                f_RR_norm = np.vstack((f_RR_norm, row))

            if features.size:
                features = np.column_stack((features, f_RR_norm))
            else:
                features = f_RR_norm

        # We have obtained RR values: both actual(3) and normalized(3) stacked vertically (51002, 6)
        # invalid beats (52536, 6)

        num_leads = np.sum(leads_flag)
        #  If both leads then 2 leads and if only MLII then 1 lead

        # Re-sampling
        if 'resample_10' in compute_morph:
            print ("Resample_10 ...")
            f_raw = np.empty((0, 10 * num_leads))
            # 10 columns or 20 columns depending on the number of leads
            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_raw_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            resamp_beat = resample(beat[s], 10)
                            if f_raw_lead.size == 1:
                                f_raw_lead = resamp_beat
                            else:
                                f_raw_lead = np.hstack((f_raw_lead, resamp_beat))
                    f_raw = np.vstack((f_raw, f_raw_lead))
            features = np.column_stack((features, f_raw)) if features.size else f_raw

        # Wavelets
        if 'wvlt' in compute_morph:
            print ("Wavelets ...")
            # 23 Features for each beat
            f_wav = np.empty((0, 23 * num_leads))
            for p in range(len(my_db.beat)):
                for b in my_db.beat[p]:
                    f_wav_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_wav_lead.size == 1:
                                f_wav_lead = compute_wavelet_descriptor(b[s], 'db1', 3)
                            else:
                                f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
                    f_wav = np.vstack((f_wav, f_wav_lead))
                    # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))

            features = np.column_stack((features, f_wav)) if features.size else f_wav

        # call other morphological features from here

        # Compute morphological features : send features and recieve features: code in features file
        # print "Computing morphological features (" + DS + ") ..."

        labels = np.array(sum(my_db.class_ID, [])).flatten()
        print("labels")

        # Set labels array!
        print('writing pickle: ' + features_labels_name + '...')
        f = open(features_labels_name, 'wb')
        pickle.dump([features, labels, patient_num_beats], f, 2)
        f.close()

    return features, labels, patient_num_beats
