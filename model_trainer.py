from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

def train_models(X_train, y_train, X_test, y_test, vectorizer=None):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[
            ('rf', models["Random Forest"]),
            ('svm', models["SVM"]),
            ('xgb', models["XGBoost"]),
            ('mlp', models["MLP"])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    models["Voting Ensemble"] = ensemble

    metrics_dict = {
        name: evaluate_model(model, X_test, y_test)
        for name, model in models.items()
    }

    return models, vectorizer, metrics_dict
