import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/baby_names.csv/names.csv")
print("Data loaded successfully!")

national = (
    df
    .groupby(["Year", "Gender", "Name"])["Births"]
    .sum()
    .reset_index()
)

print()
print("National data:")
print(national.head(20))


name = input("Enter a name to predict: ")
name_data = national[
    national["Name"].str.lower() == name.lower()
]

print()
print("Selected name:")
print(name_data.head(20))


name_data = name_data.sort_values(["Gender", "Year"])


name_data["Births_Last_Year"] = (
    name_data.groupby("Gender")["Births"].shift(1)
)

name_data["Births_Two_Years_Ago"] = (
    name_data.groupby("Gender")["Births"].shift(2)
)

print()
print("Data with historical features:")
print(name_data.head(20))

model_data = name_data.dropna().copy()

X = model_data[
    ["Year", "Gender", "Births_Last_Year", "Births_Two_Years_Ago"]
].copy()

X["Gender"] = X["Gender"].map({"F": 0, "M": 1})

y = model_data["Births"]

print()
print("Training features (X):")
print(X.head(20))

print()
print("Prediction target (y):")
print(y.head(20))


train_mask = model_data["Year"] <= 2004
test_mask = model_data["Year"] > 2004


X_train = X.loc[train_mask]
y_train = y.loc[train_mask]


X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print()
print("Training data:")
print("Years:", model_data.loc[train_mask, "Year"].min(),
      "to", model_data.loc[train_mask, "Year"].max())
print("Number of records:", len(X_train))

print()
print("Testing data:")
print("Years:", model_data.loc[test_mask, "Year"].min(),
      "to", model_data.loc[test_mask, "Year"].max())
print("Number of records:", len(X_test))


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print()
print("Model trained successfully!")