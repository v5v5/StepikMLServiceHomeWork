from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    study_features = [
        'Gender Index',
        'Age',
        'Customer Type Index',
        'Type of Travel Index',
        'Class_Business',
        'Class_Eco',
        'Flight Distance',
        'Departure Delay in Minutes',
        'Arrival Delay in Minutes',
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Inflight service',
        'Cleanliness'
    ]
    X = df[study_features]
    y = df['target']
    return X, y


def open_data(path="https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv"):
    df = pd.read_csv(path)
    # df = df[['Survived', "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):

    age_ok = df['Age'].quantile(0.997)
    df.drop(df[df['Age'] > age_ok].index, inplace=True)

    distance_ok = df['Flight Distance'].quantile(0.997)
    df.drop(df[df['Flight Distance'] > distance_ok].index, inplace=True)

    departue_delay_ok = df['Departure Delay in Minutes'].quantile(0.997)
    df.drop(df[df['Departure Delay in Minutes'] > departue_delay_ok].index, inplace=True)

    arrival_delay_ok = df['Arrival Delay in Minutes'].quantile(0.997)
    df.drop(df[df['Arrival Delay in Minutes'] > arrival_delay_ok].index, inplace=True)

    df = df[~df['satisfaction'].isin(['-'])]

    df = df.dropna()

    df['Gender Index'] = df['Gender'].map({'Male' : 1, 'Female' : 0})
    df['Customer Type Index'] = df['Customer Type'].map({'Loyal Customer' : 1, 'disloyal Customer' : 0})
    df['Type of Travel Index'] = df['Type of Travel'].map({'Business travel' : 1, 'Personal Travel' : 0})
    type_of_class = pd.get_dummies(df['Class'], prefix='Class')
    type_of_class.drop('Class_Eco Plus', axis=1, inplace=True)
    df = pd.concat([df, type_of_class], axis=1)
    df['target'] = df['satisfaction'].map({'satisfied' : 1, 'neutral or dissatisfied' : 0})

    if test:
        X_df, y_df = split_data(df)
        return X_df, y_df
    else:
        X_df = df
        return X_df        


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=42)

    # Нормализуем числовые значения
    ss = MinMaxScaler()
    ss.fit(X_train) # вычислить min, max по каждому столбцу

    X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

    # model = RandomForestClassifier()
    # model.fit(X_df, y_df)

    # test_prediction = model.predict(X_df)
    # accuracy = accuracy_score(test_prediction, y_df)
    # print(f"Model accuracy is {accuracy}")

    # with open(path, "wb") as file:
    #     dump(model, file)

    # print(f"Model was saved to {path}")

    model = LogisticRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_df)
    accuracy = accuracy_score(pred, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")



def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вам не повезло с вероятностью",
        1: "Вы выживете с вероятностью"
    }

    encode_prediction = {
        0: "Сожалеем, вам не повезло",
        1: "Ура! Вы будете жить"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)