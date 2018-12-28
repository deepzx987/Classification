# # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
# import pandas
# import numpy
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA,IncrementalPCA
# from sklearn.ensemble import ExtraTreesClassifier
#
# # load data
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
#
#  # Variance Threshold Remove features with low variance. Zero variance (same values in all samples).
#
# from sklearn.feature_selection import VarianceThreshold
# p=0.8
# sel = VarianceThreshold(threshold=(p * (1 - p)))
# new_X = sel.fit_transform(X)
# print X.shape
# print new_X.shape
#
# Univariate Selection SelectKBest removes all but the k highest scoring features SelectPercentile removes all but a
# user-specified highest scoring percentage of features using common univariate statistical tests for each feature:
# false positive rate SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.
# GenericUnivariateSelect allows to perform univariate feature selection with a configurable strategy. This allows to
# select the best univariate selection strategy with hyper-parameter search estimator. Chi squared (chi^2)
# statistical test for non-negative features to select 4 of the best features # feature extraction test =
# SelectKBest(score_func=chi2, k=4) fit = test.fit(X, Y) # summarize scores numpy.set_printoptions(precision=3) #
# print(fit.scores_) new_X = fit.transform(X) # summarize selected features # print(new_X[0:5,:]) print X.shape print
# new_X.shape
#
# ## Recursive Feature Elimination
#
# # feature extraction
# model = LogisticRegression()
# rfe = RFE(model, 3)
# fit = rfe.fit(X, Y)
# print("Num Features: %d") % fit.n_features_
# print("Selected Features: %s") % fit.support_
# print("Feature Ranking: %s") % fit.ranking_
#
# ##  Principal Component Analysis
#
# # feature extraction
# pca = PCA(n_components=3)
# fit = pca.fit(X)
# # summarize components
# print("Explained Variance: %s") % fit.explained_variance_ratio_
# print(fit.components_)
#
# ## Extra Tree Classifier
#
# # feature extraction
# model = ExtraTreesClassifier()
# model.fit(X, Y)
# print(model.feature_importances_)
#
#
#
#
#
# # Data Load
#
# import gc
# import pickle
# features_labels_name = 'w_90_90_DS1_rm_bsline_maxRR_u-lbp_MLII.p'
#
# print("Loading pickle: " + features_labels_name + "...")
# f = open(features_labels_name, 'rb')
# # disable garbage collector
# gc.disable()# this improve the required loading time!
# features, labels, patient_num_beats = pickle.load(f)
# gc.enable()
# f.close()
#
# print features.shape
# print labels.shape
# print patient_num_beats.shape
#
# # Feature Selection start...
#
# from sklearn.feature_selection import VarianceThreshold
# p=0.8
# sel = VarianceThreshold(threshold=(p * (1 - p)))
# new_features = sel.fit_transform(features)
# print features.shape
# print new_features.shape
#
# from sklearn.feature_selection import VarianceThreshold
# p=0.9
# sel = VarianceThreshold(threshold=(p * (1 - p)))
# new_features = sel.fit_transform(features)
# print features.shape
# print new_features.shape
#
# Below function me try with different k and apply models for accuracy
#
# from sklearn.neighbors import NearestNeighbors
#
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features)
#
# distances, indices = nbrs.kneighbors(features)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn import neighbors, datasets
# from sklearn import preprocessing
# from sklearn.cross_validation import train_test_split
# from sklearn import metrics
# n_neighbors = 15
#
# features_scale = preprocessing.scale(features)
#
# X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.33,random_state=17)
#
# # we create an instance of Neighbours Classifier and fit the data.
# # clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train,y_train)
#
# Z = clf.predict(X_test)
#
# print metrics.classification_report(y_test,Z)
#
#
#
# plt.plot(y_test)
# plt.plot(Z)
# plt.show()
#
# import seaborn as sns
# sns.countplot(Z,label="Count")
# plt.show()
#
# sns.countplot(y_test,label="Count")
# plt.show()
#
# #### See how scatter plot works
#
# # plt.scatter(x,y,c=c)
# # plt.show()
#
# # plt.scatter(x,y,c=c)
# # plt.show()
#
#
#
#
#
#
#
# # Chi squared (chi^2) statistical test for non-negative features to select 4 of the best features
# # feature extraction
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(features, labels)
# # summarize scores
# numpy.set_printoptions(precision=3)
# # print(fit.scores_)
# new_features = fit.transform(features)
# # summarize selected features
# # print(new_X[0:5,:])
# print features.shape
# print new_features.shape
#
#
#
# # feature extraction
# model = LogisticRegression()
# rfe = RFE(model, 3)
# fit = rfe.fit(features, labels)
# print("Num Features: %d") % fit.n_features_
# print("Selected Features: %s") % fit.support_
# print("Feature Ranking: %s") % fit.ranking_
#
#
#
#
#
# # feature extraction
# model = ExtraTreesClassifier()
# model.fit(features, labels)
# print(model.feature_importances_)
#
#
#
# # Input X_train,X_test,y_train,y_test
#
# ## PCA
#
# # feature extraction
# pca = PCA(n_components=3)
# fit = pca.fit(features)
# # summarize components
# print("Explained Variance: %s") % fit.explained_variance_ratio_
# print(fit.components_)
