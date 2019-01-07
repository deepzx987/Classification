from data_load import *

def main(multi_mode='ovo', winL=90, winR=90, do_preprocess=True, use_weight_class=True,
         maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, oversamp_method = '',
         pca_k = '', feature_selection = '', do_cross_val = '', C_value = 0.001, gamma_value = 0.0,
         reduced_DS = False, leads_flag = [1,0]):
    db_path = os.getcwd()
    print db_path

    # Load train data
    print 'LOADING TRAIN DATA'
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
                                                                 maxRR, use_RR, norm_RR, compute_morph,
                                                                 db_path, reduced_DS, leads_flag)
    print 'LOADED TRIAN DATA'