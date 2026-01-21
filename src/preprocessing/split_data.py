import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(vin_csv, covid_csv, pneu_csv, normal_csv, output_dir):
    print("Starting data splitting...")
    os.makedirs(output_dir, exist_ok=True)

    # Densenet
    print("\nPreparing Densenet data split...")
    df_vin = pd.read_csv(vin_csv)
    unique_ids = df_vin['image_id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_dense = df_vin[df_vin['image_id'].isin(train_ids)]
    val_dense = df_vin[df_vin['image_id'].isin(val_ids)]

    train_dense.to_csv(f"{output_dir}/train_dense.csv", index=False)
    val_dense.to_csv(f"{output_dir}/val_dense.csv", index=False)
    print(f"Densenet train and validation splits saved to {output_dir}")
    print(f"Train size: {len(train_dense)}, Validation size: {len(val_dense)}")

    # Alexnet
    print("\nPreparing Alexnet data split...")
    #COVID-19
    df_cov = pd.read_csv(covid_csv)
    df_cov = df_cov.rename(columns = {'FILE NAME': 'image_id'})
    df_cov = df_cov[['image_id']].copy()
    df_cov['class_name'] = 'COVID-19'
    n_covid = len(df_cov)
    print(f"Number of COVID-19 samples: {n_covid}")

    #Pneumonia
    df_pneu = pd.read_csv(pneu_csv)
    df_pneu = df_pneu.rename(columns = {'FILE NAME': 'image_id'})
    if len(df_pneu) > n_covid:
        df_pneu = df_pneu.sample(n = n_covid, random_state = 42)
    df_pneu = df_pneu[['image_id']].copy()
    df_pneu['class_name'] = 'Pneumonia'
    print(f"Number of Pneumonia samples equalized: {len(df_pneu)}")

    #Normal
    df_norm = pd.read_csv(normal_csv)
    df_norm = df_norm.rename(columns = {'FILE NAME': 'image_id'})
    if len(df_norm) > n_covid:
        df_norm = df_norm.sample(n = n_covid, random_state = 42)
    df_norm = df_norm[['image_id']].copy()
    df_norm['class_name'] = 'Normal'
    print(f"Number of Normal samples equalized: {len(df_norm)}")

    df_alex = pd.concat([df_cov, df_pneu, df_norm], ignore_index=True)

    train_alex, val_alex = train_test_split(
        df_alex, 
        test_size=0.2, 
        random_state=42, 
        stratify = df_alex['class_name']  
    )

    train_alex.to_csv(f"{output_dir}/train_alex.csv", index=False)
    val_alex.to_csv(f"{output_dir}/val_alex.csv", index=False)

    print(f"Alexnet train and validation splits saved to {output_dir}")
    print(f"Distribution in training set:\n{train_alex['class_name'].value_counts().to_dict()}")

if __name__ == "__main__":
    VIN_CSV = "data/train.csv"
    COVID_CSV = "data/covid.csv" 
    PNEU_CSV = "data/pneumonia.csv"
    NORMAL_CSV = "data/normal.csv"
    OUTPUT_DIR = "data/processed/splits"

    split_data(VIN_CSV, COVID_CSV, PNEU_CSV, NORMAL_CSV, OUTPUT_DIR)
    print("Data splitting completed.")