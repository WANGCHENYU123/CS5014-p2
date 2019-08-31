{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
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
      "                0          1          2          3          4          5    \\\n",
      "count  8.000000e+01  80.000000  80.000000  80.000000  80.000000  80.000000   \n",
      "mean   4.744000e-06  -0.003708  -0.014729  -0.021187  -0.029183  -0.031623   \n",
      "std    7.246765e-05   0.003389   0.013972   0.029048   0.050787   0.064971   \n",
      "min   -1.760000e-04  -0.007380  -0.030900  -0.060600  -0.097300  -0.120000   \n",
      "25%   -4.460000e-05  -0.007020  -0.028600  -0.051100  -0.082350  -0.100250   \n",
      "50%   -3.550000e-17  -0.004725  -0.019950  -0.033850  -0.046500  -0.048350   \n",
      "75%    5.620000e-05  -0.000267  -0.000170   0.010052   0.025950   0.038400   \n",
      "max    1.640000e-04   0.000183   0.001580   0.015100   0.036300   0.052600   \n",
      "\n",
      "             6          7          8          9    ...        758        759  \\\n",
      "count  80.000000  80.000000  80.000000  80.000000  ...  80.000000  80.000000   \n",
      "mean   -0.024276  -0.010496   0.007888   0.030303  ...   0.093090   0.100313   \n",
      "std     0.067188   0.064273   0.057468   0.047854  ...   0.047892   0.051480   \n",
      "min    -0.115000  -0.099800  -0.094700  -0.087100  ...   0.022100   0.027300   \n",
      "25%    -0.096425  -0.081075  -0.050975  -0.017900  ...   0.045900   0.049650   \n",
      "50%    -0.032550  -0.012100   0.009870   0.044900  ...   0.099800   0.104050   \n",
      "75%     0.046950   0.057200   0.064850   0.073175  ...   0.138000   0.149250   \n",
      "max     0.062300   0.072500   0.082900   0.107000  ...   0.173000   0.180000   \n",
      "\n",
      "             760        761        762        763        764        765  \\\n",
      "count  80.000000  80.000000  80.000000  80.000000  80.000000  80.000000   \n",
      "mean    0.107671   0.114716   0.119696   0.123999   0.127683   0.128745   \n",
      "std     0.055761   0.059565   0.063171   0.065070   0.065847   0.066386   \n",
      "min     0.032300   0.038100   0.043800   0.046800   0.047800   0.045700   \n",
      "25%     0.052800   0.056100   0.058150   0.061200   0.062875   0.063600   \n",
      "50%     0.108850   0.111300   0.112650   0.114450   0.123150   0.130500   \n",
      "75%     0.161250   0.173000   0.182000   0.190000   0.195000   0.196000   \n",
      "max     0.187000   0.204000   0.212000   0.209000   0.221000   0.224000   \n",
      "\n",
      "             766        767  \n",
      "count  80.000000  80.000000  \n",
      "mean    0.127553   0.125397  \n",
      "std     0.065868   0.064434  \n",
      "min     0.044000   0.042500  \n",
      "25%     0.060600   0.059975  \n",
      "50%     0.134500   0.136500  \n",
      "75%     0.194500   0.192250  \n",
      "max     0.218000   0.212000  \n",
      "\n",
      "[8 rows x 768 columns]\n"
     ]
    }
   ],
   "source": [
    "#Descriptive statistics(Binary data)\n",
    "X = pd.read_csv('binary/X.csv', header = None)\n",
    "y = pd.read_csv('binary/y.csv', header = None)\n",
    "X_descriptive = X.describe()\n",
    "y_descriptive = y.describe()\n",
    "#print('x: ',X_descriptive)\n",
    "#print('y: ',y_descriptive)\n",
    "X_around = np.around(X_descriptive, decimals = 3)\n",
    "#print(X_around)\n",
    "X_descriptive.to_csv('Results/DescriptiveStatistics/BinaryDescriptive.csv')\n",
    "#print(X_descriptive)"
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
      "Binary data\n",
      "Global min:  -0.49\n",
      "Mean min:  -0.09936791766927083\n",
      "Global max:  0.5589999999999999\n",
      "Mean max:  0.156564109765625\n",
      "Global mean:  0.029217921393717445\n"
     ]
    }
   ],
   "source": [
    "print('Binary data')\n",
    "global_min = np.min(X_descriptive.values[3, :])\n",
    "print('Global min: ', global_min)\n",
    "global_min_mean = np.mean(X_descriptive.values[3, :])\n",
    "print('Mean min: ', global_min_mean)\n",
    "global_max = np.max(X_descriptive.values[7, :])\n",
    "print('Global max: ', global_max)\n",
    "global_max_mean = np.mean(X_descriptive.values[7, :])\n",
    "print('Mean max: ', global_max_mean)\n",
    "global_mean = np.mean(X_descriptive.values[1, :])\n",
    "print('Global mean: ', global_mean)"
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
      "              0           1           2           3           4           5    \\\n",
      "count  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000   \n",
      "mean    -0.000023   -0.003385   -0.018109   -0.044432   -0.070604   -0.065050   \n",
      "std      0.000101    0.006261    0.031999    0.099239    0.181008    0.200847   \n",
      "min     -0.000335   -0.018300   -0.090700   -0.304000   -0.446000   -0.415000   \n",
      "25%     -0.000078   -0.007187   -0.029000   -0.055025   -0.089975   -0.111000   \n",
      "50%     -0.000012   -0.000459   -0.004225   -0.005685   -0.006510   -0.006720   \n",
      "75%      0.000039    0.000033   -0.000211    0.007925    0.018125    0.026800   \n",
      "max      0.000203    0.016100    0.063700    0.177000    0.390000    0.551000   \n",
      "\n",
      "              6           7           8           9    ...         758  \\\n",
      "count  200.000000  200.000000  200.000000  200.000000  ...  200.000000   \n",
      "mean    -0.051995   -0.037708   -0.023399   -0.007142  ...    0.103569   \n",
      "std      0.209133    0.216695    0.220067    0.222171  ...    0.169791   \n",
      "min     -0.389000   -0.374000   -0.358000   -0.344000  ...   -0.062500   \n",
      "25%     -0.107500   -0.095850   -0.070450   -0.058050  ...    0.001090   \n",
      "50%     -0.006250   -0.002505   -0.001855    0.001565  ...    0.045800   \n",
      "75%      0.032200    0.038000    0.044300    0.059075  ...    0.138000   \n",
      "max      0.648000    0.716000    0.761000    0.789000  ...    0.848000   \n",
      "\n",
      "              759         760         761         762         763         764  \\\n",
      "count  200.000000  200.000000  200.000000  200.000000  200.000000  200.000000   \n",
      "mean     0.115710    0.127689    0.139934    0.149543    0.158775    0.168652   \n",
      "std      0.173532    0.177492    0.181095    0.185442    0.190393    0.196031   \n",
      "min     -0.069500   -0.078600   -0.073500   -0.082200   -0.102000   -0.120000   \n",
      "25%      0.004370    0.005600    0.006918    0.003810    0.001680    0.002470   \n",
      "50%      0.053250    0.060100    0.084150    0.129000    0.142000    0.152500   \n",
      "75%      0.151250    0.167000    0.179000    0.195000    0.215750    0.247750   \n",
      "max      0.875000    0.889000    0.895000    0.896000    0.890000    0.878000   \n",
      "\n",
      "              765         766         767  \n",
      "count  200.000000  200.000000  200.000000  \n",
      "mean     0.176151    0.181042    0.184454  \n",
      "std      0.201907    0.207640    0.213561  \n",
      "min     -0.140000   -0.162000   -0.186000  \n",
      "25%      0.004260    0.006300    0.007678  \n",
      "50%      0.154000    0.168000    0.162500  \n",
      "75%      0.251500    0.264000    0.269250  \n",
      "max      0.864000    0.849000    0.836000  \n",
      "\n",
      "[8 rows x 768 columns]\n"
     ]
    }
   ],
   "source": [
    "#Descriptive statistics(Multiclass data)\n",
    "X = pd.read_csv('multiclass/X.csv', header=None)\n",
    "y = pd.read_csv('multiclass/y.csv', header=None)\n",
    "X_descriptive = X.describe()\n",
    "y_descriptive = y.describe()\n",
    "X_descriptive.to_csv(\"Results/DescriptiveStatistics/MulticlassDescriptive.csv\")\n",
    "print(X_descriptive)"
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
      "Multi class\n",
      "Global min: -0.9420000000000001\n",
      "Mean min: -0.3620581041666666\n",
      "Global max: 1.0\n",
      "Mean max: 0.53106509375\n",
      "Global mean: 0.02547285202643229\n"
     ]
    }
   ],
   "source": [
    "print('Multi class')\n",
    "global_min = np.min(X_descriptive.values[3, :])\n",
    "print(\"Global min: \" + str(global_min))\n",
    "global_min_mean = np.mean(X_descriptive.values[3, :])\n",
    "print(\"Mean min: \" + str(global_min_mean))\n",
    "global_max = np.max(X_descriptive.values[7, :])\n",
    "print(\"Global max: \" + str(global_max))\n",
    "global_max_mean = np.mean(X_descriptive.values[7, :])\n",
    "print(\"Mean max: \" + str(global_max_mean))\n",
    "gloabl_mean = np.mean(X_descriptive.values[1, :])\n",
    "print(\"Global mean: \" + str(gloabl_mean))"
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
