from sklearn.impute import KNNImputer
import pandas as pd
import os
import glob

input_dir = "ORDERED"
output_dir = "IMPUTED"

os.makedirs(output_dir, exist_ok=True)

imputer = KNNImputer(n_neighbors=5)

def impute_and_save(file):
    df = pd.read_csv(file)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    new_filename = file.replace("ordered_data", "imputed_data").replace("ordered", "imputed")
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)
    df_imputed.to_csv(new_filename, index=False)

for filepath in glob.glob(os.path.join(input_dir, "*.csv")):
    impute_and_save(filepath)
    print(f"Processed and saved: {filepath.replace('ordered_data', 'imputed_data').replace('ordered', 'imputed')}")
