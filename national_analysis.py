import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/baby_names.csv/names.csv")
national = (
    df
    .groupby(["Year", "Gender", "Name"])["Births"]
    .sum()
    .reset_index()
)

print()
print("National data:")
print(national.head(20))

name = input("Enter a name: ")

name_data = national[
    national["Name"].str.lower() == name.lower()
]
if name_data.empty:
    print()
    print("Name not found in the dataset.")
else:
    name_data = name_data.sort_values(["Year", "Gender"])

    print()
    print("Name trend over time:")
    print(
        name_data[
            ["Year", "Gender", "Births"]
        ].to_string(index=False)
    )

    most_popular = name_data.loc[
        name_data["Births"].idxmax()
    ]

    print()
    print("Most popular record:")
    print(most_popular)

    first_year = name_data["Year"].min()
    last_year = name_data["Year"].max()

    print()
    print("First year:", first_year)
    print("Last year:", last_year)
    first_year_births = name_data[
        name_data["Year"] == first_year
    ]["Births"].sum()

    last_year_births = name_data[
        name_data["Year"] == last_year
    ]["Births"].sum()

    print("First year births:", first_year_births)
    print("Last year births:", last_year_births)

    percentage_change = (
        (last_year_births - first_year_births)
        / first_year_births
    ) * 100

    print()
    print("Percentage change:", round(percentage_change, 2), "%")

    
    for gender in name_data["Gender"].unique():
        gender_data = name_data[
            name_data["Gender"] == gender
        ]

        plt.plot(
            gender_data["Year"],
            gender_data["Births"],
            label=gender
        )

    
    plt.xlabel("Year")
    plt.ylabel("Number of Births")
    plt.title(name + " Popularity Over Time")
    plt.legend()
    plt.show()