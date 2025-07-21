import sqlite3
import pandas as pd
import os

def check_csv_exists(csv_file="wallet_source_platform_dest_amount_2025_dump.csv"):
    """
    Check if the required CSV file exists, raise specific error if not.
    """
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"❌ external pull of data has query provided but not endpoint to query\n"
            f"   Missing file: {csv_file}"
        )
    
    print(f"✅ Found CSV file: {csv_file}")
    return True

def load_debridge_competitive_data(csv_file="wallet_source_platform_dest_amount_2025_dump.csv", 
                                  db_name="relay_analysis.db",
                                  table_name="across_debridge_usage"):
    """
    Load DeBridge competitive data from CSV into SQLite database.
    
    Expected CSV structure from the updated SQL query:
    - wallet_ (source_address)
    - source_chain
    - platform_ (across-v3, deBridge - with dln_debridge consolidated)
    - destination_chain  
    - amount_usd (total volume per wallet/route/platform)
    - tx_count (transaction count per wallet/route/platform)
    """
    
    # Check if CSV exists first
    check_csv_exists(csv_file)
    
    print(f"🔍 Loading DeBridge competitive data from {csv_file}...")
    print(f"📅 Data period: 2025-01-01 to 2025-07-01 (same 6-month window as Relay)")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df):,} rows from CSV")
        
        # Validate expected columns
        expected_cols = ['wallet_', 'source_chain', 'platform_', 'destination_chain', 'amount_usd', 'tx_count']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing expected columns: {missing_cols}")
        
        # Preview the data structure
        print(f"\n📋 CSV Structure:")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Shape: {df.shape}")
        
        # Show competitive platforms
        if 'platform_' in df.columns:
            platforms = df['platform_'].value_counts()
            print(f"\n🏢 Competitive Platforms:")
            for platform, count in platforms.items():
                print(f"   {platform}: {count:,} records")
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\n⚠️  Missing values found:")
            for col, count in missing_summary[missing_summary > 0].items():
                print(f"   {col}: {count:,} missing")
        
        # Connect to database
        conn = sqlite3.connect(db_name)
        
        try:
            # Drop existing table if it exists
            conn.execute(f"DROP TABLE IF EXISTS {table_name};")
            
            # Insert data into SQLite
            print(f"\n💾 Inserting data into {table_name} table...")
            df.to_sql(table_name, conn, index=False, if_exists='replace')
            
            # Create indexes for faster analysis
            if 'wallet_' in df.columns:
                conn.execute(f"CREATE INDEX idx_{table_name}_wallet ON {table_name}(wallet_);")
                print(f"✅ Created index on wallet_ column")
            
            if 'platform_' in df.columns:
                conn.execute(f"CREATE INDEX idx_{table_name}_platform ON {table_name}(platform_);")
                print(f"✅ Created index on platform_ column")
            
            if 'source_chain' in df.columns and 'destination_chain' in df.columns:
                conn.execute(f"CREATE INDEX idx_{table_name}_route ON {table_name}(source_chain, destination_chain);")
                print(f"✅ Created index on route (source_chain, destination_chain)")
            
            # Get row count to verify
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            row_count = result[0]
            
            print(f"✅ Successfully inserted {row_count:,} rows into {table_name}")
            
            return df
            
        finally:
            conn.close()
            
    except pd.errors.EmptyDataError:
        raise ValueError(f"❌ CSV file {csv_file} is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"❌ Error parsing CSV file {csv_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading data to database: {e}")

if __name__ == "__main__":
    print("🚀 Starting DeBridge competitive data import...")
    
    try:
        # Load competitive data from CSV to database
        df = load_debridge_competitive_data()
        
        print(f"\n🎯 SUMMARY")
        print("=" * 60)
        print(f"✅ Data import completed")
        print(f"📊 Table created: across_debridge_usage")
        print(f"📈 Data ready for analysis")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        exit(1) 