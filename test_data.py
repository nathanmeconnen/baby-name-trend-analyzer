import pandas as pd

df = pd.read_csv("data/baby_names.csv/names.csv")

print("First 5 rows:")
print(df.head())

print()

print("Column names:")
print(df.columns.tolist())

print()

print("Dataset size:")
print(df.shape)

print()

print("Missing values:")
print(df.isnull().sum())

print()

print("Year range:")
print(df["Year"].min(), "to", df["Year"].max())

print()

print("Number of states:")
print(df["State"].nunique())

print()

print("States:")
print(df["State"].unique())

print()

print("Genders:")
print(df["Gender"].unique())

print()

print("Number of unique names:")
print(df["Name"].nunique())

print()

print("Olivia:")
print(df[df["Name"] == "Olivia"].head(20))
