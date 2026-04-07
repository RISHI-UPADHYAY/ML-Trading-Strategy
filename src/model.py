from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler