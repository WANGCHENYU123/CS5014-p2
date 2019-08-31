{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "import Function\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import f_classif\n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(200, 768)\n",
      "(200,)\n",
      "(50, 768)\n"
     ]
    }
   ],
   "source": [
    "#Load data\n",
    "X = np.loadtxt(open('multiclass/X.csv'), delimiter = \",\")\n",
    "y = np.loadtxt(open('multiclass/y.csv'), delimiter = \",\")\n",
    "X_classify = np.loadtxt(open('multiclass/XToClassify.csv'), delimiter = \",\")\n",
    "print(X.shape)\n",
    "print(y.shape)\n",
    "print(X_classify.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(160, 768)\n",
      "(40, 768)\n",
      "(160,)\n",
      "(40,)\n"
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
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the Number of samples for each class\n",
      "(40,)\n",
      "(40,)\n",
      "(40,)\n",
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
    "print(y1)\n",
    "y2 = y[y == 2].shape\n",
    "print(y2)\n",
    "y3 = y[y == 3].shape\n",
    "print(y3)\n",
    "y4 = y[y == 4].shape\n",
    "print(y4)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 11  0  0  1]\n",
      " [ 0  1  5  0  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
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
    "Function.write_file(y_predict_result_LR, 'Results/Multiclass/LogRegression.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_LR)\n",
    "print(cm)\n",
    "#classification rate:95%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SelectKBest\n",
      "[50]\n"
     ]
    }
   ],
   "source": [
    "#SelectKBest(k = 1)\n",
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PCA\n",
      "(160, 10)\n",
      "(40, 10)\n",
      "(50, 10)\n"
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
    "print(X_PCA_train.shape)\n",
    "X_PCA_test = pca.transform(X_test)\n",
    "print(X_PCA_test.shape)\n",
    "X_PCA_classify = pca.transform(X_classify)\n",
    "print(X_PCA_classify.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression with PCA\n",
      "[[ 5  0  0  0  0]\n",
      " [ 2 10  0  0  0]\n",
      " [ 0  0  4  1  1]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
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
    "Function.write_file(y_predict_result_PCA_LR, 'Results/Multiclass/PCALR.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_PCA_LR)\n",
    "print(cm)\n",
    "#90%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression with SelectKBest\n",
      "[[ 5  0  0  0  0]\n",
      " [11  0  1  0  0]\n",
      " [ 1  0  4  1  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  7  0  0]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
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
    "Function.write_file(y_predict_result_best_LR, 'Results/Multiclass/SelctBestLR.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_best_LR)\n",
    "print(cm)\n",
    "#47.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SelectKBest\n",
      "[48, 49, 50, 51, 52, 53, 562, 563, 564, 565]\n"
     ]
    }
   ],
   "source": [
    "#SelectKBest(k = 10)\n",
    "print(\"SelectKBest\")\n",
    "selector10 = SelectKBest(f_classif, k = 10)\n",
    "selector10.fit(X_train, y_train)\n",
    "#the score of each feature\n",
    "selector.scores_\n",
    "#print(\"scores:\",selector.scores_)\n",
    "#the column that get the highest score\n",
    "X_best_train10 = selector10.transform(X_train)\n",
    "X_best_test10 = selector10.transform(X_test)\n",
    "X_best_classify10 = selector10.transform(X_classify)\n",
    "#get the index of the column\n",
    "mask = selector10._get_support_mask()\n",
    "print(Function.cal_index(mask, True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic regression with SelectKBest(10)\n",
      "[[ 5  0  0  0  0]\n",
      " [ 8  0  0  0  4]\n",
      " [ 2  0  2  1  1]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n",
      "/usr/local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.\n",
      "  \"this warning.\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#Logistic regression with SelectKBest(10)\n",
    "print(\"Logistic regression with SelectKBest(10)\")\n",
    "LR = LogisticRegression()\n",
    "LR.fit(X_best_train10, y_train)\n",
    "y_predict_test_best10_LR = LR.predict(X_best_test10)\n",
    "y_predict_result_best10_LR = LR.predict(X_best_classify10)\n",
    "Function.write_file(y_predict_result_best10_LR, 'Results/Multiclass/SelctBest10LR.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_best10_LR)\n",
    "print(cm)\n",
    "#classification rate:60%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 12  0  0  0]\n",
      " [ 0  2  4  0  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
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
    "Function.write_file(y_predict_result_svc, 'Results/Multiclass/SVC.csv')\n",
    "Function.write_file(y_predict_result_svc, 'MultiClassPredictedClass/PredictedClasses.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_svc)\n",
    "print(cm)\n",
    "#classification rate:95%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC with Best K Selection\n",
      "[[ 5  0  0  0  0]\n",
      " [12  0  0  0  0]\n",
      " [ 4  0  1  1  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 7  0  0  0  0]]\n"
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
    "Function.write_file(y_predict_result_best_svc, 'Results/Multiclass/SelctBestSVC.csv')\n",
    "#print(confusion_matrix(y_test, y_predict_test_best_svc))\n",
    "cm = confusion_matrix(y_test, y_predict_test_best_svc)\n",
    "print(cm)\n",
    "#classification rate:40%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC with Best K Selection(10)\n",
      "[[ 5  0  0  0  0]\n",
      " [12  0  0  0  0]\n",
      " [ 3  0  0  1  2]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "# SVC with SelectKBest(10)\n",
    "print(\"SVC with Best K Selection(10)\")\n",
    "svc = SVC(kernel='linear', C=1)\n",
    "svc_model = svc.fit(X_best_train10, y_train)\n",
    "y_predict_test_best10_svc = svc.predict(X_best_test10)\n",
    "y_predict_result_best10_svc = svc.predict(X_best_classify10)\n",
    "Function.write_file(y_predict_result_best10_svc, 'Results/Multiclass/SelctBest10SVC.csv')\n",
    "#print(confusion_matrix(y_test, y_predict_test_best_svc))\n",
    "cm = confusion_matrix(y_test, y_predict_test_best10_svc)\n",
    "print(cm)\n",
    "#classification rate:55%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC with PCA\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 12  0  0  0]\n",
      " [ 0  2  4  0  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "# SVC with PCA\n",
    "print(\"SVC with PCA\")\n",
    "svc = SVC(kernel='linear', C=1)\n",
    "svc_model = svc.fit(X_PCA_train, y_train)\n",
    "y_predict_test_PCA_svc = svc.predict(X_PCA_test)\n",
    "y_predict_result_PCA_svc = svc.predict(X_PCA_classify)\n",
    "Function.write_file(y_predict_result_PCA_svc, 'Results/Multiclass/PCASVC.csv')\n",
    "#print(confusion_matrix(y_test, y_predict_test_best_svc))\n",
    "cm = confusion_matrix(y_test, y_predict_test_PCA_svc)\n",
    "print(cm)\n",
    "#classification rate:95%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Multi-Layer Perception(layer_size(5,5))\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 12  0  0  0]\n",
      " [ 1  1  4  0  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception laryer_size(5,5)\n",
    "print(\"Multi-Layer Perception(layer_size(5,5))\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)\n",
    "mlp.fit(X_train, y_train)\n",
    "y_predict_test_mlp = mlp.predict(X_test)\n",
    "y_predict_result_mlp = mlp.predict(X_classify)\n",
    "Function.write_file(y_predict_result_mlp, 'Results/Multiclass/MLPLaryer5.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_mlp)\n",
    "print(cm)\n",
    "#classification rate:95%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Multi-Layer Perception(layer_size(7,7))\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 12  0  0  0]\n",
      " [ 0  0  6  0  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception laryer_size(7,7)\n",
    "print(\"Multi-Layer Perception(layer_size(7,7))\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(8, 8), random_state=1)\n",
    "mlp.fit(X_train, y_train)\n",
    "y_predict_test_mlp = mlp.predict(X_test)\n",
    "y_predict_result_mlp = mlp.predict(X_classify)\n",
    "Function.write_file(y_predict_result_mlp, 'Results/Multiclass/MLPLayer7.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_mlp)\n",
    "print(cm)\n",
    "#classification rate:100%"
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
      "Multi-Layer Perception(layer_size(7,7)) with SelectKBest\n",
      "[[ 5  0  0  0  0]\n",
      " [ 2  9  1  0  0]\n",
      " [ 0  1  2  1  2]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception with selectKBest laryer_size(7,7)\n",
    "print(\"Multi-Layer Perception(layer_size(7,7)) with SelectKBest\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7, 7), random_state=1)\n",
    "mlp.fit(X_best_train, y_train)\n",
    "y_predict_test_best_mlp = mlp.predict(X_best_test)\n",
    "y_predict_result_best_mlp = mlp.predict(X_best_classify)\n",
    "Function.write_file(y_predict_result_best_mlp, 'Results/Multiclass/SelctBestMLPLayer7.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_best_mlp)\n",
    "print(cm)\n",
    "#classification rate:82.5%"
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
      "Multi-Layer Perception(layer_size(7,7)) with SelectKBest\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 11  1  0  0]\n",
      " [ 0  2  4  0  0]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception with selectKBest(10) laryer_size(7,7)\n",
    "print(\"Multi-Layer Perception(layer_size(7,7)) with SelectKBest\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7, 7), random_state=1)\n",
    "mlp.fit(X_best_train10, y_train)\n",
    "y_predict_test_best10_mlp = mlp.predict(X_best_test10)\n",
    "y_predict_result_best10_mlp = mlp.predict(X_best_classify10)\n",
    "Function.write_file(y_predict_result_best10_mlp, 'Results/Multiclass/SelctBest10MLPLayer7.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_best10_mlp)\n",
    "print(cm)\n",
    "#classification rate:95%"
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
      "Multi-Layer Perception(layer_size(7,7)) with PCA\n",
      "[[ 5  0  0  0  0]\n",
      " [ 0 12  0  0  0]\n",
      " [ 0  0  4  1  1]\n",
      " [ 0  0  0 10  0]\n",
      " [ 0  0  0  0  7]]\n"
     ]
    }
   ],
   "source": [
    "#Multi-Layer Perception with PCA laryer_size(7,7)\n",
    "print(\"Multi-Layer Perception(layer_size(7,7)) with PCA\")\n",
    "mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7, 7), random_state=1)\n",
    "mlp.fit(X_PCA_train, y_train)\n",
    "y_predict_test_PCA_mlp = mlp.predict(X_PCA_test)\n",
    "y_predict_result_PCA_mlp = mlp.predict(X_PCA_classify)\n",
    "Function.write_file(y_predict_result_PCA_mlp, 'Results/Multiclass/PCAMLPLayer7.csv')\n",
    "cm = confusion_matrix(y_test, y_predict_test_PCA_mlp)\n",
    "print(cm)\n",
    "#classification rate:95%"
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
