{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "def write_file(list, file_name):\n",
    "    file = open(file_name, 'w')\n",
    "    for i in list:\n",
    "        file.write(\"%s\\n\" % int(i))\n",
    "\n",
    "def cal_index(array, number):\n",
    "    index = 0\n",
    "    indices = []\n",
    "    for i in array:\n",
    "        if i == number:\n",
    "            indices = indices + [index]\n",
    "        index = index + 1\n",
    "    return indices\n",
    "\n",
    "\n",
    "def evaluation_metrices(con_matrix):\n",
    "    classification_rate = (con_matrix[0, 0] + con_matrix[1, 1])/(con_matrix[0, 0] + con_matrix[1, 1] + con_matrix[1, 0] + con_matrix[0, 1])\n",
    "    specificity = 1 - (con_matrix[1, 0]/(con_matrix[1, 0] + con_matrix[1, 1]))\n",
    "    sensitivity = 1 - (con_matrix[0, 1] / (con_matrix[0, 0] + con_matrix[0, 1]))\n",
    "    precision = (con_matrix[0, 0] / (con_matrix[0, 0] + con_matrix[1, 0]))\n",
    "    recall = (con_matrix[0, 0] / (con_matrix[0, 0] + con_matrix[0, 1]))\n",
    "    print('Classification Rate: ' + str(classification_rate))\n",
    "    print('Specificity: ' + str(specificity))\n",
    "    print('Sensitivity: ' + str(sensitivity))\n",
    "    print('Precision: ' + str(precision))\n",
    "    print('Recall: ' + str(recall))\n",
    "    return [classification_rate, specificity, sensitivity, precision, recall]\n",
    "\n"
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
