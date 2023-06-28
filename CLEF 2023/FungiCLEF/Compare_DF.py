import pandas as pd

# Carga el primer archivo en un dataframe
df1 = pd.read_csv("Results/Convnext_Base_All_95.csv")

# Carga el segundo archivo en un dataframe
df2 = pd.read_csv("Results/Convnext_Base_All_95_ENB7_Base_All_95.csv")

# Une los dataframes por la columna 'observation_id'
df = pd.merge(df1, df2, on="observation_id")

# Renombra las columnas para que reflejen los nombres de archivo originales
df.rename(
    columns={"class_id_x": "class_id_file1", "class_id_y": "class_id_file2"},
    inplace=True,
)

# Muestra el dataframe resultante
print(df)

num_diff = (df["class_id_file1"] != df["class_id_file2"]).sum()

# Imprime el número de filas diferentes
print(
    f"Hay {num_diff} filas donde las columnas class_id_file1 y class_id_file2 son diferentes"
)
