{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.70      0.87      0.77        69\n",
      "           1       0.70      0.45      0.55        47\n",
      "\n",
      "    accuracy                           0.70       116\n",
      "   macro avg       0.70      0.66      0.66       116\n",
      "weighted avg       0.70      0.70      0.68       116\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "df_heart = pd.read_csv('SAHeart.csv', index_col=0)\n",
    "df_heart.head(10)\n",
    "df_heart.describe()\n",
    "# df_heart.drop('famhist', axis=1, inplace=True)\n",
    "df_heart = pd.get_dummies(df_heart, columns = ['famhist'], drop_first=True)\n",
    "# Set random seed\n",
    "seed = 52\n",
    "# Split into train and test sections\n",
    "y = df_heart.pop('chd')\n",
    "X_train, X_test, y_train, y_test = train_test_split(df_heart, y, test_size=0.25, random_state=seed)\n",
    "# Build logistic regression model\n",
    "model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)\n",
    "# Report training set score\n",
    "train_score = model.score(X_train, y_train) * 100\n",
    "# Report test set score\n",
    "test_score = model.score(X_test, y_test) * 100\n",
    "# Write scores to a file\n",
    "with open(\"metrics.txt\", 'w') as outfile:\n",
    "        outfile.write(\"Training variance explained: %2.1f%%n\" % train_score)\n",
    "        outfile.write(\"Test variance explained: %2.1f%%n\" % test_score)\n",
    "# Confusion Matrix and plot\n",
    "cm = confusion_matrix(y_test, model.predict(X_test))\n",
    "fig, ax = plt.subplots(figsize=(8, 8))\n",
    "ax.imshow(cm)\n",
    "ax.grid(False)\n",
    "ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))\n",
    "ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))\n",
    "ax.set_ylim(1.5, -0.5)\n",
    "for i in range(2):\n",
    "    for j in range(2):\n",
    "        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"cm.png\",dpi=120) \n",
    "plt.close()\n",
    "# Print classification report\n",
    "print(classification_report(y_test, model.predict(X_test)))\n",
    "#roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])\n",
    "# Plot the ROC curve\n",
    "model_ROC = plot_roc_curve(model, X_test, y_test)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"roc.png\",dpi=120) \n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
