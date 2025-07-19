import pandas as pd
import glob
from pathlib import Path

def explore_relay_csvs():
    """
    Read first 5 rows from each relay CSV file and compare column schemas.
    Similar to R's head() function for quick data exploration.
    """
    
    # Get all relay CSV files
    csv_files = sorted(glob.glob("relay_*.csv"))
    
    if not csv_files:
        print("No relay CSV files found in current directory")
        return
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    print("=" * 60)
    
    # Store data and column info for comparison
    dataframes = {}
    all_columns = {}
    
    # Read first 5 rows from each file
    for file in csv_files:
        try:
            print(f"\n📁 Reading {file}...")
            df = pd.read_csv(file, nrows=5)  # Like head(5) in R
            dataframes[file] = df
            all_columns[file] = list(df.columns)
            
            print(f"Shape: {df.shape}")
            print(f"Columns ({len(df.columns)}): {list(df.columns)}")
            print("\nFirst 5 rows:")
            print(df.to_string(index=False))
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
    
    # Combine all 5-row samples into one CSV
    if dataframes:
        print(f"\n💾 WRITING COMBINED SAMPLE CSV")
        print("=" * 60)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes.values(), ignore_index=True)
        
        # Write to new CSV file
        output_file = "relay_combined_sample.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"✅ Combined sample written to: {output_file}")
        print(f"Combined shape: {combined_df.shape}")
        print(f"Total rows from {len(dataframes)} files: {combined_df.shape[0]} rows")
    
    # Compare column schemas across files
    print("\n🔍 COLUMN SCHEMA COMPARISON")
    print("=" * 60)
    
    if len(all_columns) > 1:
        # Get reference columns (from first file)
        reference_file = list(all_columns.keys())[0]
        reference_cols = set(all_columns[reference_file])
        
        print(f"Reference file: {reference_file}")
        print(f"Reference columns ({len(reference_cols)}): {sorted(reference_cols)}")
        print()
        
        schema_consistent = True
        
        for file, cols in all_columns.items():
            if file == reference_file:
                continue
                
            current_cols = set(cols)
            
            if current_cols == reference_cols:
                print(f"✅ {file}: MATCHING schema")
            else:
                schema_consistent = False
                print(f"❌ {file}: DIFFERENT schema")
                
                missing = reference_cols - current_cols
                extra = current_cols - reference_cols
                
                if missing:
                    print(f"   Missing columns: {sorted(missing)}")
                if extra:
                    print(f"   Extra columns: {sorted(extra)}")
        
        print(f"\n📊 Schema consistency: {'✅ CONSISTENT' if schema_consistent else '❌ INCONSISTENT'}")
        
        # Show column counts summary
        print(f"\nColumn count summary:")
        for file, cols in all_columns.items():
            print(f"  {file}: {len(cols)} columns")
    
    else:
        print("Only one file found - no comparison possible")

if __name__ == "__main__":
    explore_relay_csvs() 