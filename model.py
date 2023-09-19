from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import wooldridge as wd 
import pickle

data = wd.data('mroz')

y = data['inlf']
X = data[['kidslt6', 'age', 'educ', 'huswage']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pg_tree = {'max_depth' : [3, 4, 5]}
pg_rf = {'max_depth' : [3, 4, 5],
           'n_estimators': [50, 100, 150] }

gs_tree = GridSearchCV(DecisionTreeClassifier(),
                        cv=3, param_grid=pg_tree)
gs_rf = GridSearchCV(RandomForestClassifier(), 
                        cv=3, param_grid=pg_rf)

gs_tree.fit(X_train, y_train)
print(gs_tree.best_score_)
gs_rf.fit(X_train, y_train)
print(gs_rf.best_score_)

rf_pred = gs_tree.predict(X_test)
tree_pred = gs_rf.predict(X_test)

as_rf = accuracy_score(rf_pred, y_test) 
as_t = accuracy_score(tree_pred, y_test)

if as_rf > as_t:
    with open('model.pkl', 'wb') as f:
        pickle.dump(gs_rf, f)
else:
    with open('model.pkl', 'wb') as f:
        pickle.dump(gs_tree, f)


