import os

import pandas as pd


# Function to update lotes dropdown menus
def update_lote_dropdown_options():
    lotes_df = load_lotes()
    if not lotes_df.empty:
        options = [f"{row['Nombre del Campo']} ({row['Identificación del Lote']})" for _, row in lotes_df.iterrows()]
    else:
        options = ["No hay lotes disponibles"]
    return options

# Function to load lotes from the CSV file
def load_lotes():
    csv_folder = "csvs"
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, "lotes.csv")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=["Nombre del Campo", "Identificación del Lote", "Archivo"])

# Function to save lotes to the CSV file
def save_lotes(df):
    csv_folder = "csvs"
    csv_path = os.path.join(csv_folder, "lotes.csv")
    df.to_csv(csv_path, index=False)
    

# Function to load ambientes from the CSV file
def load_ambientes():
    csv_folder = "csvs"
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, "ambientes.csv")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=["Orden", "Ambiente"])

# Function to save ambientes to the CSV file
def save_ambientes(df):
    csv_folder = "csvs"
    csv_path = os.path.join(csv_folder, "ambientes.csv")
    df.to_csv(csv_path, index=False)