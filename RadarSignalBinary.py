{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import Function\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import f_classif"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(80, 768)\n",
      "(80,)\n",
      "(20, 768)\n"
     ]
    }
   ],
   "source": [
    "#Load data\n",
    "X = np.loadtxt(open('binary/X.csv'), delimiter = \",\")\n",
    "y = np.loadtxt(open('binary/y.csv'), delimiter = \",\")\n",
    "X_classify = np.loadtxt(open('binary/XToClassify.csv'), delimiter = \",\")\n",
    "print(X.shape)\n",
    "print(y.shape)\n",
    "print(X_classify.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(64, 768)\n",
      "(16, 768)\n",
      "(64,)\n",
      "(16,)\n"
     ]
    }
   ],
   "source": [
    "#split into trainning set and test set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)\n",
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the Number of samples for each class\n",
      "(40,)\n",
      "(40,)\n"
     ]
    }
   ],
   "source": [
    "print(\"the Number of samples for each class\")\n",
    "y0 = y[y == 0].shape\n",
    "print(y0)\n",
    "y1 = y[y == 1].shape\n",
    "print(y1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SelectKBest\n",
      "[26]\n"
     ]
    }
   ],
   "source": [
    "#SelectKBest\n",
    "print(\"SelectKBest\")\n",
    "selector = SelectKBest(f_classif, k = 1)\n",
    "selector.fit(X_train, y_train)\n",
    "#the score of each feature\n",
    "selector.scores_\n",
    "#print(\"scores:\",selector.scores_)\n",
    "#the column that get the highest score\n",
    "X_best_train = selector.transform(X_train)\n",
    "X_best_test = selector.transform(X_test)\n",
    "X_best_classify = selector.transform(X_classify)\n",
    "#get the index of the column\n",
    "mask = selector._get_support_mask()\n",
    "print(Function.cal_index(mask, True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PCA\n"
     ]
    }
   ],
   "source": [
    "#PCA choose 10 features\n",
    "print(\"PCA\")\n",
    "pca = PCA(n_components = 10)\n",
    "pca.fit(X_train)\n",
    "X_PCA_train = pca.transform(X_train)\n",
    "#print(X_pca_train)\n",
    "X_PCA_test = pca.transform(X_test)\n",
    "X_PCA_classify = pca.transform(X_classify)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Multi-Layer Perception\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception\n",
    "print(\"Multi-Layer Perception\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)\n",
    "mlp.fit(X_train, y_train)\n",
    "y_predict_test_mlp = mlp.predict(X_test)\n",
    "y_predict_result_mlp = mlp.predict(X_classify)\n",
    "Function.write_file(y_predict_result_mlp, 'Results/Binary/MLP.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_mlp)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Multi-Layer Perception\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception with SelectKBest\n",
    "print(\"Multi-Layer Perception\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)\n",
    "mlp.fit(X_best_train, y_train)\n",
    "y_predict_test_best_mlp = mlp.predict(X_best_test)\n",
    "y_predict_result_best_mlp = mlp.predict(X_best_classify)\n",
    "#Function.write_file(y_predict_result_best_mlp, 'Results/Binary/SelctBestMLP.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_best_mlp)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Multi-Layer Perception with PCA\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception with PCA\n",
    "print(\"Multi-Layer Perception with PCA\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)\n",
    "mlp.fit(X_PCA_train, y_train)\n",
    "y_predict_test_PCA_mlp = mlp.predict(X_PCA_test)\n",
    "y_predict_result_PCA_mlp = mlp.predict(X_PCA_classify)\n",
    "#Function.write_file(y_predict_result_PCA_mlp, 'Results/Binary/PCAMLP.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_PCA_mlp)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#Logistic Regression\n",
    "print(\"Logistic regression\")\n",
    "LR = LogisticRegression()\n",
    "LR.fit(X_train, y_train)\n",
    "y_predict_test_LR = LR.predict(X_test)\n",
    "y_predict_result_LR = LR.predict(X_classify)\n",
    "Function.write_file(y_predict_result_LR, 'Results/Binary/LogRegression.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_LR)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression with SelectKBest\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#Logistic regression with SelectKBest\n",
    "print(\"Logistic regression with SelectKBest\")\n",
    "LR = LogisticRegression()\n",
    "LR.fit(X_best_train, y_train)\n",
    "y_predict_test_best_LR = LR.predict(X_best_test)\n",
    "y_predict_result_best_LR = LR.predict(X_best_classify)\n",
    "Function.write_file(y_predict_result_best_LR, 'Results/Binary/SelctBestLR.csv')\n",
    "Function.write_file(y_predict_result_best_LR, 'BinaryPredictedClass/PredictedClasses.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_best_LR)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression with PCA\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#Logistic regression with PCA\n",
    "print(\"Logistic regression with PCA\")\n",
    "LR = LogisticRegression()\n",
    "LR.fit(X_PCA_train, y_train)\n",
    "y_predict_test_PCA_LR = LR.predict(X_PCA_test)\n",
    "y_predict_result_PCA_LR = LR.predict(X_PCA_classify)\n",
    "Function.write_file(y_predict_result_PCA_LR, 'Results/Binary/PCALogisticRegre')\n",
    "cm = confusion_matrix(y_test, y_predict_test_PCA_LR)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression with Recursive feature selection\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[28, 29, 30, 284, 285, 286, 540, 541, 656, 657]\n",
      "[[11  0]\n",
      " [ 0  5]]\n",
      "(768,)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#Logistic Regression with Recursive feature selection\n",
    "print(\"Logistic regression with Recursive feature selection\")\n",
    "LR = LogisticRegression()\n",
    "rfe = RFE(estimator = LR, n_features_to_select = 10, step = 10)\n",
    "rfe.fit(X_train, y_train)\n",
    "y_predict_test_RFE_LR = rfe.predict(X_test)\n",
    "y_predict_result_RFE_LR = rfe.predict(X_classify)\n",
    "Function.write_file(y_predict_result_RFE_LR, 'Results/Binary/RFELogisticRegre.csv')\n",
    "#10 Features are chosen\n",
    "print(Function.cal_index(rfe.support_, True))\n",
    "cm = confusion_matrix(y_test, y_predict_test_RFE_LR)\n",
    "print(cm)\n",
    "ranking = rfe.ranking_\n",
    "print(ranking.shape)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "# SVC\n",
    "print(\"SVC\")\n",
    "svc = SVC(kernel='linear', C=1)\n",
    "svc_model = svc.fit(X_train, y_train)\n",
    "y_predict_test_svc = svc.predict(X_test)\n",
    "y_predict_result_svc = svc.predict(X_classify)\n",
    "Function.write_file(y_predict_result_svc, 'Results/Binary/SVC.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_svc)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC with Best K Selection\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "# SVC with SelectKBest\n",
    "print(\"SVC with Best K Selection\")\n",
    "svc = SVC(kernel='linear', C=1)\n",
    "svc_model = svc.fit(X_best_train, y_train)\n",
    "y_predict_test_best_svc = svc.predict(X_best_test)\n",
    "y_predict_result_best_svc = svc.predict(X_best_classify)\n",
    "Function.write_file(y_predict_result_best_svc, 'Results/Binary/SelctBestSVC.csv')\n",
    "#print(confusion_matrix(y_test, y_predict_test_best_svc))\n",
    "cm = confusion_matrix(y_test, y_predict_test_best_svc)\n",
    "print(cm)\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recursive Feature Selection and SVC\n",
      "[21, 22, 23, 278, 532, 533, 534, 535, 536, 537]\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "# SVC with Recursive feature selection \n",
    "print(\"Recursive Feature Selection and SVC\")\n",
    "svc = SVC(kernel=\"linear\", C=1)\n",
    "rfe = RFE(estimator=svc, n_features_to_select=10, step=100)\n",
    "rfe.fit(X_train, y_train)\n",
    "# print(rfe.support_)\n",
    "y_predict_test_rfe_svc = rfe.predict(X_test)\n",
    "y_predict_result_rfe_svc = rfe.predict(X_classify)\n",
    "Function.write_file(y_predict_result_rfe_svc, 'Results/Binary/RFESVC.csv')\n",
    "print(Function.cal_index(rfe.support_, True))\n",
    "cm = confusion_matrix(y_test, y_predict_test_rfe_svc)\n",
    "print(cm)\n",
    "ranking = rfe.ranking_\n",
    "#classification rate:100%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC with PCA\n",
      "[[11  0]\n",
      " [ 0  5]]\n"
     ]
    }
   ],
   "source": [
    "# SVC with PCA\n",
    "print(\"SVC with PCA\")\n",
    "svc = SVC(kernel='linear', C=1)\n",
    "svc_model = svc.fit(X_PCA_train, y_train)\n",
    "y_predict_test_pca_svc = svc.predict(X_PCA_test)\n",
    "y_predict_result_pca_svc = svc.predict(X_PCA_classify)\n",
    "Function.write_file(y_predict_result_pca_svc, 'Results/Binary/PCASVC.csv')\n",
    "#print(confusion_matrix(y_test, y_pred_test_pca_svc))\n",
    "cm = confusion_matrix(y_test, y_predict_test_pca_svc)\n",
    "print(cm)\n",
    "# print(new_components.shape)\n",
    "#classification rate:100%"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
