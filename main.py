from data_load import *


def main(multi_mode='ovo', winL=90, winR=90, do_preprocess=True, use_weight_class=True,
         maxRR=True, use_RR=True, norm_RR=True, compute_morph={''}, oversamp_method='',
         pca_k='', feature_selection='', do_cross_val='', C_value=0.001, gamma_value=0.0,
         reduced_DS=False, leads_flag=[1, 0]):
    """

    :string compute_morph: all the types of features u want to calculate
    """
    db_path = os.getcwd()
    print(db_path)

    # Load train data
    print('LOADING TRAIN DATA')
    [tr_features, tr_labels, tr_patient_num_beats] = load_mit_db('DS1', winL, winR, do_preprocess,
                                                                 maxRR, use_RR, norm_RR, compute_morph,
                                                                 db_path, reduced_DS, leads_flag)
    print('LOADED TRAIN DATA')

    # Load test data
    print('LOADING TEST DATA')
    [eval_features, eval_labels, eval_patient_num_beats] = load_mit_db('DS2', winL, winR, do_preprocess,
                                                                       maxRR, use_RR, norm_RR, compute_morph, db_path,
                                                                       reduced_DS, leads_flag)
    print('LOADED TEST DATA')


winL = 90
winR = 90
do_preprocess = True
use_weight_class = True
maxRR = True
compute_morph = {'resample_10', 'wvlt'}  # 'wvlt', 'HOS', 'myMorph'
multi_mode = 'ovo'
voting_strategy = 'ovo_voting'  # 'ovo_voting_exp', 'ovo_voting_both'
use_RR = False
norm_RR = False
oversamp_method = ''
feature_selection = ''
do_cross_val = ''
C_value = 0.001
reduced_DS = False  # To select only patients in common with MLII and V1
# Will use both the leads: it will create redundant data but it will also increase the instances of small classes.
leads_flag = [1, 1]  # MLII, V1
pca_k = '0'
ov_methods = {''}  # , 'SMOTE_regular'}
C_values = {0.001, 0.01, 0.1, 1, 10, 100}
gamma_values = {0.0}
gamma_value = 0.0
main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k,
     feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
