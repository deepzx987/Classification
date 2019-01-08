from data_load import *


def main(multi_mode='ovo', winL=90, winR=90, do_preprocess=True, use_weight_class=True,
         maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, oversamp_method='',
         pca_k='', feature_selection='', do_cross_val='', C_value=0.001, gamma_value=0.0,
         reduced_DS=False, leads_flag=[1, 0]):
    db_path = os.getcwd()
    print db_path

    # Load train data
    print 'LOADING TRAIN DATA'
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
                                                                 maxRR, use_RR, norm_RR, compute_morph,
                                                                 db_path, reduced_DS, leads_flag)
    print 'LOADED TRIAN DATA'


winL = 90
winR = 90
do_preprocess = True
use_weight_class = True
maxRR = True
compute_morph = {''}  # 'wvlt', 'HOS', 'myMorph'

multi_mode = 'ovo'
voting_strategy = 'ovo_voting'  # 'ovo_voting_exp', 'ovo_voting_both'

use_RR = False
norm_RR = False

oversamp_method = ''
feature_selection = ''
do_cross_val = ''
C_value = 0.001
reduced_DS = False  # To select only patients in common with MLII and V1
leads_flag = [1, 0]  # MLII, V1

pca_k = 0

################

# With feature selection
ov_methods = {''}  # , 'SMOTE_regular'}

C_values = {0.001, 0.01, 0.1, 1, 10, 100}
gamma_values = {0.0}
gamma_value = 0.0
main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k,
     feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
