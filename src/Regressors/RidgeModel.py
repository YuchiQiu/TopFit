from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import utils
from mlde_utils import merge_tuple

REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]
reg_unsupervised=10**-8

def RidgeRegressor(X_train,Y_train,X_test,n_cv=5,num_structure=None):
    # concatenate features if they are given in tuple format
    X_train=merge_tuple(X_train,reg_unsupervised)
    X_test=merge_tuple(X_test,reg_unsupervised)
    best_rc, best_score = None, -np.inf
    for rc in REG_COEF_LIST:
        model = Ridge(alpha=rc)
        kf = KFold(n_splits=n_cv)
        score = []
        if len(X_train.shape) == 2:
            for train_index, test_index in kf.split(X_train):
                x_train, x_test = X_train[train_index, :], X_train[test_index, :]
                y_train, y_test = Y_train[train_index], Y_train[test_index]
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score.append(utils.spearman(y_pred, y_test))
        elif len(X_train.shape) == 3:
            if num_structure is None:
                num_structure=X_train.shape[1]
            for train_index, test_index in kf.split(X_train[:, 0, :]):
                y_train, y_test = Y_train[train_index], Y_train[test_index]
                y_pred = []
                for model_id in range(num_structure):
                    x_train, x_test = X_train[train_index, model_id, :], X_train[test_index, model_id, :]
                    model.fit(x_train, y_train)
                    y_pred.append(model.predict(x_test))
                y_pred = np.mean(np.asarray(y_pred), axis=0)
                score.append(utils.spearman(y_pred, y_test))
        score=np.asarray(score).mean()
        if score > best_score:
            best_rc = rc
            best_score = score
    if len(X.shape)==2:
        model = Ridge(alpha=best_rc)
        model.fit(X_train, Y_train)
        preds=model.predict(X_test)
    elif len(X.shape) == 3:
        model=[]
        preds=[]
        for model_id in range(num_structure):
            model.append(Ridge(alpha=best_rc))
            model[model_id].fit(X_train[:,model_id,:],Y_train)
            preds.append(model[model_id].predict(X_test[:,model_id,:]))
        preds=np.asarray(preds)
        preds=np.mean(preds,axis=0)

    top_models_name=['Ridge']
    top_models_params={'alpha':best_rc}
    return preds,top_models_name,top_models_params,training_loss_cv,testing_loss_cv

def train(self, train_seqs, train_labels,optimize_round=1,num_ensemble=5):

    X=np.append(X1,X2,axis=len(X1.shape)-1)
    best_rc, best_score = None, -np.inf
        for rc in REG_COEF_LIST:
            model = self.linear_model_cls(alpha=rc)
            # CV for alpha selection;
            # if X.shape==3: NMR model is used. X.shape[0] is the feature from specific NMR structure model
            # we predict fitness on each model and ensemble them for CV scoring.
            kf = KFold(n_splits=5)
            score=[]
            if len(X.shape)==2:
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index,:], X[test_index,:]
                    y_train, y_test = train_labels[train_index], train_labels[test_index]
                    model.fit(X_train,y_train)
                    y_pred=model.predict(X_test)
                    score.append(utils.spearman(y_pred, y_test))
            elif len(X.shape)==3:
                for train_index, test_index in kf.split(X[:,0,:]):
                    y_train, y_test = train_labels[train_index], train_labels[test_index]
                    y_pred=[]
                    for model_id in range(X.shape[1]):
                        X_train, X_test = X[train_index,model_id,:], X[test_index,model_id,:]
                        model.fit(X_train, y_train)
                        y_pred.append(model.predict(X_test))
                    y_pred=np.mean(np.asarray(y_pred),axis=0)
                    score.append(utils.spearman(y_pred, y_test))
            score=np.asarray(score).mean()
            # score = cross_val_score(
            #     model, X, train_labels, cv=5,
            #     scoring=make_scorer(utils.spearman)).mean()
            if score > best_score:
                best_rc = rc
                best_score = score
        self.reg_coef = best_rc
        # print(f'Cross validated reg coef {best_rc}')
    if len(X.shape)==2:
        self.model = self.linear_model_cls(alpha=self.reg_coef)
        self.model.fit(X, train_labels)
    elif len(X.shape) == 3:
        self.model=[]
        for model_id in range(X.shape[1]):
            self.model.append(self.linear_model_cls(alpha=self.reg_coef))
            self.model[model_id].fit(X[:,model_id,:],train_labels)

    return self.reg_coef