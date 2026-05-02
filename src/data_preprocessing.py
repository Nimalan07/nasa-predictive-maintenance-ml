import pandas as pd
import os

RAW_PATH = "data/raw/train_FD001.txt"
PROCESSED_PATH = "data/processed/processed_train.csv"


def load_data():
  
    df = pd.read_csv(RAW_PATH, sep=" ", header=None)
    
    df = df.dropna(axis=1)
    
    columns = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    df.columns = columns
    
    return df


def add_rul(df):
    
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    
    
    df = df.merge(max_cycle, on='engine_id')
    
   
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    return df


def create_target(df, threshold=30):

    df['failure'] = df['RUL'].apply(lambda x: 1 if x <= threshold else 0)
    return df


def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f" Processed data saved to {PROCESSED_PATH}")


def main():
    print("Loading data...")
    df = load_data()
    
    print("Creating RUL...")
    df = add_rul(df)
    
    print("Creating target variable...")
    df = create_target(df)
    
    print("Saving processed data...")
    save_data(df)


if __name__ == "__main__":
    main()