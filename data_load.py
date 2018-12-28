def load_mit_db(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag):
    print("Runing train_SVM.py!")

    features_labels_name = create_features_labels_name(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR,
                                                       compute_morph, db_path, reduced_DS, leads_flag)

    if os.path.isfile(features_labels_name):
        print("Loading pickle: " + features_labels_name + "...")
        f = open(features_labels_name, 'rb')
        # disable garbage collector
        gc.disable()  # this improve the required loading time!
        features, labels, patient_num_beats = pickle.load(f)
        gc.enable()
        f.close()


    else:
        print("Loading MIT BIH arr (" + DS + ") ...")

        # ML-II
        if reduced_DS == False:
            DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                   223, 230]
            DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                   233, 234]

        # ML-II + V1
        else:
            DS1 = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
            DS2 = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

        db_path = os.getcwd()
        mit_pickle_name = db_path + '/python_mit'

        print mit_pickle_name

        if reduced_DS:
            mit_pickle_name = mit_pickle_name + '_reduced_'

        if do_preprocess:
            mit_pickle_name = mit_pickle_name + '_rm_bsline'

        print mit_pickle_name

        mit_pickle_name = mit_pickle_name + '_wL_' + str(winL) + '_wR_' + str(winR)
        print mit_pickle_name

        mit_pickle_name = mit_pickle_name + '_' + DS + '.p'

        print mit_pickle_name

        # If the data with that configuration has been already computed Load pickle
        if os.path.isfile(mit_pickle_name):
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
            # Save data
            # Protocol version 0 itr_features_balanceds the original ASCII protocol and is backwards compatible with earlier versions of Python.
            # Protocol version 1 is the old binary format which is also compatible with earlier versions of Python.
            # Protocol version 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes.
            f = open(mit_pickle_name, 'wb')
            pickle.dump(my_db, f, 2)
            f.close

        features = np.array([], dtype=float)
        labels = np.array([], dtype=np.int32)

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
                    RR[p] = compute_RR_intervals(my_db.R_pos[p])
                else:
                    RR[p] = compute_RR_intervals(my_db.orig_R_pos[p])

                RR[p].pre_R = RR[p].pre_R[(my_db.valid_R[p] == 1)]
                RR[p].post_R = RR[p].post_R[(my_db.valid_R[p] == 1)]
                RR[p].local_R = RR[p].local_R[(my_db.valid_R[p] == 1)]
                RR[p].global_R = RR[p].global_R[(my_db.valid_R[p] == 1)]

        if use_RR:
            f_RR = np.empty((0, 4))
            for p in range(len(RR)):
                row = np.column_stack((RR[p].pre_R, RR[p].post_R, RR[p].local_R, RR[p].global_R))
                f_RR = np.vstack((f_RR, row))

            features = np.column_stack((features, f_RR)) if features.size else f_RR

        if norm_RR:
            f_RR_norm = np.empty((0, 4))
            for p in range(len(RR)):
                # Compute avg values!
                avg_pre_R = np.average(RR[p].pre_R)
                avg_post_R = np.average(RR[p].post_R)
                avg_local_R = np.average(RR[p].local_R)
                avg_global_R = np.average(RR[p].global_R)

                row = np.column_stack((RR[p].pre_R / avg_pre_R, RR[p].post_R / avg_post_R, RR[p].local_R / avg_local_R,
                                       RR[p].global_R / avg_global_R))
                f_RR_norm = np.vstack((f_RR_norm, row))

            features = np.column_stack((features, f_RR_norm)) if features.size else f_RR_norm

        #########################################################################################
        # Compute morphological features
        print("Computing morphological features (" + DS + ") ...")

        num_leads = np.sum(leads_flag)

        # Raw
        if 'resample_10' in compute_morph:
            print("Resample_10 ...")
            start = time.time()

            f_raw = np.empty((0, 10 * num_leads))

            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_raw_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            resamp_beat = scipy.signal.resample(beat[s], 10)
                            if f_raw_lead.size == 1:
                                f_raw_lead = resamp_beat
                            else:
                                f_raw_lead = np.hstack((f_raw_lead, resamp_beat))
                    f_raw = np.vstack((f_raw, f_raw_lead))

            features = np.column_stack((features, f_raw)) if features.size else f_raw

            end = time.time()
            print("Time resample: " + str(format(end - start, '.2f')) + " sec")

        if 'raw' in compute_morph:
            print("Raw ...")
            start = time.time()

            f_raw = np.empty((0, (winL + winR) * num_leads))

            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_raw_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_raw_lead.size == 1:
                                f_raw_lead = beat[s]
                            else:
                                f_raw_lead = np.hstack((f_raw_lead, beat[s]))
                    f_raw = np.vstack((f_raw, f_raw_lead))

            features = np.column_stack((features, f_raw)) if features.size else f_raw

            end = time.time()
            print("Time raw: " + str(format(end - start, '.2f')) + " sec")
        # LBP 1D
        # 1D-local binary pattern based feature extraction for classification of epileptic EEG signals: 2014, unas 55 citas, Q2-Q1 Matematicas
        # https://ac.els-cdn.com/S0096300314008285/1-s2.0-S0096300314008285-main.pdf?_tid=8a8433a6-e57f-11e7-98ec-00000aab0f6c&acdnat=1513772341_eb5d4d26addb6c0b71ded4fd6cc23ed5

        # 1D-LBP method, which derived from implementation steps of 2D-LBP, was firstly proposed by Chatlani et al. for detection of speech signals that is non-stationary in nature [23]

        # From Raw signal

        # TODO: Some kind of preprocesing or clean high frequency noise?

        # Compute 2 Histograms: LBP or Uniform LBP
        # LBP 8 = 0-255
        # U-LBP 8 = 0-58
        # Uniform LBP are only those pattern wich only presents 2 (or less) transitions from 0-1 or 1-0
        # All the non-uniform patterns are asigned to the same value in the histogram

        if 'u-lbp' in compute_morph:
            print("u-lbp ...")

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

        if 'lbp' in compute_morph:
            print("lbp ...")

            f_lbp = np.empty((0, 16 * num_leads))

            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_lbp_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_lbp_lead.size == 1:

                                f_lbp_lead = compute_LBP(beat[s], 4)
                            else:
                                f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(beat[s], 4)))
                    f_lbp = np.vstack((f_lbp, f_lbp_lead))

            features = np.column_stack((features, f_lbp)) if features.size else f_lbp
            print(features.shape)

        if 'hbf5' in compute_morph:
            print("hbf ...")

            f_hbf = np.empty((0, 15 * num_leads))

            for p in range(len(my_db.beat)):
                for beat in my_db.beat[p]:
                    f_hbf_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_hbf_lead.size == 1:

                                f_hbf_lead = compute_HBF(beat[s])
                            else:
                                f_hbf_lead = np.hstack((f_hbf_lead, compute_HBF(beat[s])))
                    f_hbf = np.vstack((f_hbf, f_hbf_lead))

            features = np.column_stack((features, f_hbf)) if features.size else f_hbf
            print(features.shape)

        # Wavelets
        if 'wvlt' in compute_morph:
            print("Wavelets ...")

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

        # Wavelets
        if 'wvlt+pca' in compute_morph:
            pca_k = 7
            print("Wavelets + PCA (" + str(pca_k) + "...")

            family = 'db1'
            level = 3

            f_wav = np.empty((0, 23 * num_leads))

            for p in range(len(my_db.beat)):
                for b in my_db.beat[p]:
                    f_wav_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_wav_lead.size == 1:
                                f_wav_lead = compute_wavelet_descriptor(b[s], family, level)
                            else:
                                f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], family, level)))
                    f_wav = np.vstack((f_wav, f_wav_lead))
                    # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))

            if DS == 'DS1':
                # Compute PCA
                # PCA = sklearn.decomposition.KernelPCA(pca_k) # gamma_pca
                IPCA = IncrementalPCA(n_components=pca_k,
                                      batch_size=10)  # NOTE: due to memory errors, we employ IncrementalPCA
                IPCA.fit(f_wav)

                # Save PCA
                save_wvlt_PCA(IPCA, pca_k, family, level)
            else:
                # Load PCAfrom sklearn.decomposition import PCA, IncrementalPCA
                IPCA = load_wvlt_PCA(pca_k, family, level)
            # Extract the PCA
            # f_wav_PCA = np.empty((0, pca_k * num_leads))
            f_wav_PCA = IPCA.transform(f_wav)
            features = np.column_stack((features, f_wav_PCA)) if features.size else f_wav_PCA

        # HOS
        if 'HOS' in compute_morph:
            print("HOS ...")
            n_intervals = 6
            lag = int(round((winL + winR) / n_intervals))

            f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
            for p in range(len(my_db.beat)):
                for b in my_db.beat[p]:
                    f_HOS_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_HOS_lead.size == 1:
                                f_HOS_lead = compute_hos_descriptor(b[s], n_intervals, lag)
                            else:
                                f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
                    f_HOS = np.vstack((f_HOS, f_HOS_lead))
                    # f_HOS = np.vstack((f_HOS, compute_hos_descriptor(b, n_intervals, lag)))

            features = np.column_stack((features, f_HOS)) if features.size else f_HOS
            print(features.shape)

        # My morphological descriptor
        if 'myMorph' in compute_morph:
            print("My Descriptor ...")
            f_myMorhp = np.empty((0, 4 * num_leads))
            for p in range(len(my_db.beat)):
                for b in my_db.beat[p]:
                    f_myMorhp_lead = np.empty([])
                    for s in range(2):
                        if leads_flag[s] == 1:
                            if f_myMorhp_lead.size == 1:
                                f_myMorhp_lead = compute_my_own_descriptor(b[s], winL, winR)
                            else:
                                f_myMorhp_lead = np.hstack(
                                    (f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
                    f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
                    # f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))

            features = np.column_stack((features, f_myMorhp)) if features.size else f_myMorhp

        labels = np.array(sum(my_db.class_ID, [])).flatten()
        print("labels")

        # Set labels array!
        print('writing pickle: ' + features_labels_name + '...')
        f = open(features_labels_name, 'wb')
        pickle.dump([features, labels, patient_num_beats], f, 2)
        f.close

    return features, labels, patient_num_beats
