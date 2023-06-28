import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("FungiCLEF2023_train_metadata_PRODUCTION.csv")
train_df["image_path"] = train_df["image_path"].str.lower()
train_df.to_csv("FungiCLEF2023_train_metadata_PRODUCTION_lower.csv", index=False)
print(train_df)

val_df = pd.read_csv("FungiCLEF2023_val_metadata_PRODUCTION.csv")
print(val_df)

test_df = pd.read_csv("FungiCLEF2023_public_test_metadata_PRODUCTION.csv")
print(test_df)

train_species = pd.unique(train_df["class_id"])
print(train_species)
print(min(train_species))
print(max(train_species))
print(len(train_species))

val_species = pd.unique(val_df["class_id"])
print(val_species)
print(min(val_species))
print(max(val_species))
print(len(val_species))

train_counts = train_df["class_id"].value_counts(sort=True)
print(train_counts)

df = pd.DataFrame()
df["Count"] = train_counts
df["Class"] = range(len(train_counts))
print(df["Count"].describe())
plt.scatter(df["Class"], df["Count"])
plt.xlabel("Species")
plt.ylabel("Images")
plt.show()
