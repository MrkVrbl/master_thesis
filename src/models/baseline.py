from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torchmetrics import AUROC, Accuracy
from utils.utils import get_X_y


def baseline(train_df, test_df, max_iter=200):

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    baseline = LogisticRegression(max_iter=max_iter, random_state=42)

    # flatten the data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    baseline.fit(X_train_flat, y_train)

    y_train = torch.tensor(y_train.values.astype(int))
    y_test = torch.tensor(y_test.values.astype(int))


    baseline_pred_train = torch.from_numpy(baseline.predict(X_train_flat)).int()
    baseline_pred_test = torch.from_numpy(baseline.predict(X_test_flat)).int()

    auroc = AUROC(num_classes=1)
    acc = Accuracy()

    #train_acc_score = acc(y_train, baseline_pred_train)
    test_acc_score = acc(y_test, baseline_pred_test)

    #train_auc_score = auroc(y_train, baseline_pred_train)
    test_auc_score = auroc(y_test, baseline_pred_test)

    return  test_acc_score, test_auc_score, baseline


def baseline_sklearn(train_df, test_df, max_iter=200):
    
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    baseline = LogisticRegression(max_iter=max_iter)

    # flatten the data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    baseline.fit(X_train_flat, y_train)

    # AUC
    #baseline_prob_train = baseline.predict_proba(X_train_flat)[:,1]
    baseline_prob_test = baseline.predict_proba(X_test_flat)[:,1]

    #train_auc_score = roc_auc_score(y_train, baseline_prob_train)
    test_auc_score = roc_auc_score(y_test, baseline_prob_test)

    # accuracy
    #baseline_pred_train = baseline.predict(X_train_flat)
    baseline_pred_test = baseline.predict(X_test_flat)

    #train_acc_score = accuracy_score(y_train, baseline_pred_train)
    test_acc_score = accuracy_score(y_test, baseline_pred_test)

    return  test_acc_score, test_auc_score, baseline