from classes import *


def compute_rr_intervals(R_poses):
    features_RR = RR_intervals()
    # pre_R = np.array([], dtype=int)
    post_R = np.array([], dtype=int)
    local_R = np.array([], dtype=int)
    global_R = np.array([], dtype=int)
    # Pre_R and Post_R
    # pre_R = np.append(pre_R, 0)
    post_R = np.append(post_R, R_poses[1] - R_poses[0])
    for i in range(1, len(R_poses) - 1):
        # pre_R = np.append(pre_R, R_poses[i] - R_poses[i - 1])
        post_R = np.append(post_R, R_poses[i + 1] - R_poses[i])
    # pre_R[0] = pre_R[1]
    # pre_R = np.append(pre_R, R_poses[-1] - R_poses[-2])
    post_R = np.append(post_R, post_R[-1])
    # Local_R: AVG from last 10 pre_R values
    for i in range(0, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j + i >= 0:
                # avg_val = avg_val + pre_R[i + j]
                avg_val = avg_val + post_R[i + j]
                num = num + 1
        local_R = np.append(local_R, avg_val / float(num))
    # Global R AVG: from full past-signal
    # global_R = np.append(global_R, pre_R[0])
    global_R = np.append(global_R, post_R[0])
    for i in range(1, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(0, i):
            if (R_poses[i] - R_poses[j]) < 108000:
                # avg_val = avg_val + pre_R[j]
                avg_val = avg_val + post_R[j]
                num = num + 1
        # num = i
        global_R = np.append(global_R, avg_val / float(num))
    for i in range(0, len(R_poses)):
        # features_RR.pre_R = np.append(features_RR.pre_R, pre_R[i])
        features_RR.post_R = np.append(features_RR.post_R, post_R[i])
        features_RR.local_R = np.append(features_RR.local_R, local_R[i])
        features_RR.global_R = np.append(features_RR.global_R, global_R[i])

        # features_RR.append([pre_R[i], post_R[i], local_R[i], global_R[i]])
    return features_RR


# num_leads = np.sum(leads_flag)

# Raw
# if 'resample_10' in compute_morph:
#     print("Resample_10 ...")
#     start = time.time()
#
#     f_raw = np.empty((0, 10 * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for beat in my_db.beat[p]:
#             f_raw_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     resamp_beat = scipy.signal.resample(beat[s], 10)
#                     if f_raw_lead.size == 1:
#                         f_raw_lead = resamp_beat
#                     else:
#                         f_raw_lead = np.hstack((f_raw_lead, resamp_beat))
#             f_raw = np.vstack((f_raw, f_raw_lead))
#
#     features = np.column_stack((features, f_raw)) if features.size else f_raw
#
#     end = time.time()
#     print("Time resample: " + str(format(end - start, '.2f')) + " sec")
#
# if 'raw' in compute_morph:
#     print("Raw ...")
#     start = time.time()
#
#     f_raw = np.empty((0, (winL + winR) * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for beat in my_db.beat[p]:
#             f_raw_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_raw_lead.size == 1:
#                         f_raw_lead = beat[s]
#                     else:
#                         f_raw_lead = np.hstack((f_raw_lead, beat[s]))
#             f_raw = np.vstack((f_raw, f_raw_lead))
#
#     features = np.column_stack((features, f_raw)) if features.size else f_raw
#
#     end = time.time()
#     print("Time raw: " + str(format(end - start, '.2f')) + " sec")
# # LBP 1D
# # 1D-local binary pattern based feature extraction for classification of epileptic EEG signals: 2014, unas 55 citas, Q2-Q1 Matematicas
# # https://ac.els-cdn.com/S0096300314008285/1-s2.0-S0096300314008285-main.pdf?_tid=8a8433a6-e57f-11e7-98ec-00000aab0f6c&acdnat=1513772341_eb5d4d26addb6c0b71ded4fd6cc23ed5
# # 1D-LBP method, which derived from implementation steps of 2D-LBP, was firstly proposed by Chatlani et al. for detection of speech signals that is non-stationary in nature [23]
#
# # From Raw signal
# # Compute 2 Histograms: LBP or Uniform LBP
# # LBP 8 = 0-255
# # U-LBP 8 = 0-58
# # Uniform LBP are only those pattern wich only presents 2 (or less) transitions from 0-1 or 1-0
# # All the non-uniform patterns are asigned to the same value in the histogram
#
# if 'u-lbp' in compute_morph:
#     print("u-lbp ...")
#
#     f_lbp = np.empty((0, 59 * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for beat in my_db.beat[p]:
#             f_lbp_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_lbp_lead.size == 1:
#
#                         f_lbp_lead = compute_Uniform_LBP(beat[s], 8)
#                     else:
#                         f_lbp_lead = np.hstack((f_lbp_lead, compute_Uniform_LBP(beat[s], 8)))
#             f_lbp = np.vstack((f_lbp, f_lbp_lead))
#
#     features = np.column_stack((features, f_lbp)) if features.size else f_lbp
#     print(features.shape)
#
# if 'lbp' in compute_morph:
#     print("lbp ...")
#
#     f_lbp = np.empty((0, 16 * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for beat in my_db.beat[p]:
#             f_lbp_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_lbp_lead.size == 1:
#
#                         f_lbp_lead = compute_LBP(beat[s], 4)
#                     else:
#                         f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(beat[s], 4)))
#             f_lbp = np.vstack((f_lbp, f_lbp_lead))
#
#     features = np.column_stack((features, f_lbp)) if features.size else f_lbp
#     print(features.shape)
#
# if 'hbf5' in compute_morph:
#     print("hbf ...")
#
#     f_hbf = np.empty((0, 15 * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for beat in my_db.beat[p]:
#             f_hbf_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_hbf_lead.size == 1:
#
#                         f_hbf_lead = compute_HBF(beat[s])
#                     else:
#                         f_hbf_lead = np.hstack((f_hbf_lead, compute_HBF(beat[s])))
#             f_hbf = np.vstack((f_hbf, f_hbf_lead))
#
#     features = np.column_stack((features, f_hbf)) if features.size else f_hbf
#     print(features.shape)
#
# # Wavelets
# if 'wvlt' in compute_morph:
#     print("Wavelets ...")
#
#     f_wav = np.empty((0, 23 * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for b in my_db.beat[p]:
#             f_wav_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_wav_lead.size == 1:
#                         f_wav_lead = compute_wavelet_descriptor(b[s], 'db1', 3)
#                     else:
#                         f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
#             f_wav = np.vstack((f_wav, f_wav_lead))
#             # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))
#
#     features = np.column_stack((features, f_wav)) if features.size else f_wav
#
# # Wavelets
# if 'wvlt+pca' in compute_morph:
#     pca_k = 7
#     print("Wavelets + PCA (" + str(pca_k) + "...")
#
#     family = 'db1'
#     level = 3
#
#     f_wav = np.empty((0, 23 * num_leads))
#
#     for p in range(len(my_db.beat)):
#         for b in my_db.beat[p]:
#             f_wav_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_wav_lead.size == 1:
#                         f_wav_lead = compute_wavelet_descriptor(b[s], family, level)
#                     else:
#                         f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], family, level)))
#             f_wav = np.vstack((f_wav, f_wav_lead))
#             # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))
#
#     if DS == 'DS1':
#         # Compute PCA
#         # PCA = sklearn.decomposition.KernelPCA(pca_k) # gamma_pca
#         IPCA = IncrementalPCA(n_components=pca_k,
#                               batch_size=10)  # NOTE: due to memory errors, we employ IncrementalPCA
#         IPCA.fit(f_wav)
#
#         # Save PCA
#         save_wvlt_PCA(IPCA, pca_k, family, level)
#     else:
#         # Load PCAfrom sklearn.decomposition import PCA, IncrementalPCA
#         IPCA = load_wvlt_PCA(pca_k, family, level)
#     # Extract the PCA
#     # f_wav_PCA = np.empty((0, pca_k * num_leads))
#     f_wav_PCA = IPCA.transform(f_wav)
#     features = np.column_stack((features, f_wav_PCA)) if features.size else f_wav_PCA
#
# # HOS
# if 'HOS' in compute_morph:
#     print("HOS ...")
#     n_intervals = 6
#     lag = int(round((winL + winR) / n_intervals))
#
#     f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
#     for p in range(len(my_db.beat)):
#         for b in my_db.beat[p]:
#             f_HOS_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_HOS_lead.size == 1:
#                         f_HOS_lead = compute_hos_descriptor(b[s], n_intervals, lag)
#                     else:
#                         f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
#             f_HOS = np.vstack((f_HOS, f_HOS_lead))
#             # f_HOS = np.vstack((f_HOS, compute_hos_descriptor(b, n_intervals, lag)))
#
#     features = np.column_stack((features, f_HOS)) if features.size else f_HOS
#     print(features.shape)
#
# # My morphological descriptor
# if 'myMorph' in compute_morph:
#     print("My Descriptor ...")
#     f_myMorhp = np.empty((0, 4 * num_leads))
#     for p in range(len(my_db.beat)):
#         for b in my_db.beat[p]:
#             f_myMorhp_lead = np.empty([])
#             for s in range(2):
#                 if leads_flag[s] == 1:
#                     if f_myMorhp_lead.size == 1:
#                         f_myMorhp_lead = compute_my_own_descriptor(b[s], winL, winR)
#                     else:
#                         f_myMorhp_lead = np.hstack(
#                             (f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
#             f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
#             # f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))
#
#     features = np.column_stack((features, f_myMorhp)) if features.size else f_myMorhp

# since only valid beats were put into class ID hence the number of instances of features is equal to labels/outputs
