import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("DF20-train_metadata.csv")
print(train_df)

val_df = pd.read_csv("DF20-val_metadata.csv")
print(val_df)

test_df = pd.read_csv("FungiCLEF2022_test_metadata.csv")
print(test_df)

train_species = pd.unique(train_df['class_id'])
print(train_species)
print(len(train_species))

val_species = pd.unique(val_df['class_id'])
print(val_species)
print(len(val_species))

train_counts = train_df['class_id'].value_counts(sort=True)
print(train_counts)

df = pd.DataFrame()
df['Count'] = train_counts
df['Class'] = range(len(train_counts))
print(df['Count'].describe())
plt.scatter(df['Class'], df['Count'])
plt.xlabel("Species")
plt.ylabel("Images")
plt.show()