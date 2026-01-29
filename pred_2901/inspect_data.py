import pandas as pd
from pathlib import Path

# CONFIG: Point this to your dataset folder
DATASET_PATH = Path('/home/kulkarni/projects/pred_2901/dataset')

def inspect_one_file():
    # 1. Find a file
    normal_path = DATASET_PATH / '0'
    files = list(normal_path.rglob("*.parquet"))
    
    if not files:
        print("ERROR: No .parquet files found in 'dataset/0/'")
        return

    f = files[0]
    print(f"Inspecting file: {f}")
    
    try:
        df = pd.read_parquet(f)
        
        print("\n--- 1. SHAPE ---")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        if len(df) < 120:
            print("❌ FAIL: File is shorter than 120 rows (Window Size).")
        else:
            print("✅ PASS: File is long enough.")

        print("\n--- 2. COLUMN NAMES ---")
        print(df.columns.tolist())
        
        # Test the Logic
        cols = [c for c in df.columns if ('P-' in c or 'T-' in c or 'Q' in c) 
                and 'ESTADO' not in c and 'state' not in c.lower()]
        
        print(f"\nSelected Sensors ({len(cols)}): {cols}")
        if len(cols) == 0:
            print("❌ FAIL: No columns matched 'P-', 'T-', or 'Q'. Check naming convention.")
        else:
            print("✅ PASS: Found sensor columns.")

        print("\n--- 3. DATA PREVIEW (First 5 rows) ---")
        print(df.head())
        
        # Check NaNs
        print("\n--- 4. MISSING VALUES ---")
        print(df.isna().sum())

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_one_file()