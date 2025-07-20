import sqlite3
import pandas as pd
import glob
from pathlib import Path

def create_relay_database(db_name="relay_analysis.db"):
    """
    Create SQLite database for Relay transaction analysis.
    Processes relay_*.csv files with known column schema.
    """
    
    # Expected columns (in order)
    expected_columns = [
        'created_at', 'wallet', 'execution_kind', 'is_call', 'route_source',
        'deposit_tx_hash', 'origin_chain_name', 'user_send_currency', 
        'user_send_currency_amount', 'user_send_currency_usd',
        'destination_chain_name', 'fill_tx_hash', 'user_receive_currency_amount',
        'user_receive_currency_usd', 'user_receive_currency'
    ]
    
    # Get all relay CSV files
    csv_files = sorted([f for f in glob.glob("relay_*.csv") 
                       if not f.startswith("relay_combined")])
    
    if not csv_files:
        print("❌ No relay CSV files found")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files: {csv_files}")
    print(f"🗄️  Creating SQLite database: {db_name}")
    print("=" * 70)
    
    # Remove existing database to start fresh
    db_path = Path(db_name)
    if db_path.exists():
        db_path.unlink()
        print(f"🗑️  Removed existing database")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    
    try:
        total_rows = 0
        table_created = False
        
        for i, file in enumerate(csv_files, 1):
            print(f"\n📊 Processing {file} [{i}/{len(csv_files)}]")
            
            try:
                # Get file size for progress tracking
                with open(file, 'r', encoding='utf-8') as f:
                    total_file_rows = sum(1 for _ in f) - 1  # -1 for header
                print(f"   📏 File size: {total_file_rows:,} rows")
                
                # Process in chunks to avoid SQL variable limits
                chunk_size = 1000
                file_rows_inserted = 0
                
                # Read CSV in chunks with error handling
                csv_reader = pd.read_csv(
                    file,
                    chunksize=chunk_size,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    low_memory=False
                )
                
                for chunk_num, df_chunk in enumerate(csv_reader, 1):
                    
                    # Validate columns on first chunk
                    if not table_created:
                        print(f"   🔍 Validating columns...")
                        actual_columns = list(df_chunk.columns)
                        
                        if actual_columns != expected_columns:
                            print(f"   ⚠️  Column mismatch detected!")
                            print(f"   Expected: {expected_columns}")
                            print(f"   Actual:   {actual_columns}")
                            
                            # Check if columns match but in different order
                            if set(actual_columns) == set(expected_columns):
                                print(f"   🔧 Same columns, reordering...")
                                df_chunk = df_chunk[expected_columns]
                            else:
                                missing = set(expected_columns) - set(actual_columns)
                                extra = set(actual_columns) - set(expected_columns)
                                if missing:
                                    print(f"   ❌ Missing columns: {missing}")
                                if extra:
                                    print(f"   ❌ Extra columns: {extra}")
                                raise ValueError("Column schema mismatch")
                        
                        print(f"   ✅ Column validation passed")
                    
                    # Insert chunk into SQLite
                    if_exists = 'replace' if not table_created else 'append'
                    
                    df_chunk.to_sql(
                        name='relay_transactions',
                        con=conn,
                        if_exists=if_exists,
                        index=False,
                        method='multi'
                    )
                    
                    table_created = True
                    file_rows_inserted += len(df_chunk)
                    
                    # Progress update for large files
                    if chunk_num % 100 == 0:
                        print(f"   📊 Processed {file_rows_inserted:,}/{total_file_rows:,} rows...")
                
                total_rows += file_rows_inserted
                print(f"   ✅ Successfully inserted {file_rows_inserted:,} rows")
                
            except UnicodeDecodeError:
                print(f"   ⚠️  UTF-8 encoding failed, trying latin1...")
                try:
                    file_rows_inserted = 0
                    csv_reader = pd.read_csv(
                        file,
                        chunksize=chunk_size,
                        encoding='latin1',
                        on_bad_lines='skip'
                    )
                    
                    for df_chunk in csv_reader:
                        if_exists = 'replace' if not table_created else 'append'
                        df_chunk.to_sql('relay_transactions', conn, if_exists=if_exists, index=False)
                        table_created = True
                        file_rows_inserted += len(df_chunk)
                    
                    total_rows += file_rows_inserted
                    print(f"   ✅ Recovered with latin1: {file_rows_inserted:,} rows")
                    
                except Exception as e2:
                    print(f"   ❌ Failed with latin1: {e2}")
                    continue
                    
            except pd.errors.EmptyDataError:
                print(f"   ⚠️  File appears empty, skipping...")
                continue
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        if not table_created:
            print("❌ No data was successfully loaded")
            return
        
        # Filter data to 2025+ only (drop rows before Jan 1, 2025)
        print(f"\n🗑️  Filtering data to 2025+ only...")
        
        # Count rows before filtering
        cursor = conn.execute("SELECT COUNT(*) FROM relay_transactions")
        rows_before = cursor.fetchone()[0]
        
        # Delete rows before 2025-01-01
        conn.execute("DELETE FROM relay_transactions WHERE created_at < '2025-01-01'")
        
        # Count rows after filtering
        cursor = conn.execute("SELECT COUNT(*) FROM relay_transactions")
        rows_after = cursor.fetchone()[0]
        
        rows_deleted = rows_before - rows_after
        print(f"   ✅ Deleted {rows_deleted:,} pre-2025 rows ({rows_after:,} rows remaining)")
        
        # Create indexes for analysis performance
        print(f"\n🔧 Creating indexes for faster queries...")
        
        indexes = [
            ("idx_created_at", "created_at"),
            ("idx_wallet", "wallet"), 
            ("idx_origin_chain", "origin_chain_name"),
            ("idx_dest_chain", "destination_chain_name"),
            ("idx_execution_kind", "execution_kind"),
            ("idx_route_source", "route_source")
        ]
        
        for idx_name, column in indexes:
            try:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON relay_transactions({column})")
                print(f"   ✅ Created index on {column}")
            except Exception as e:
                print(f"   ⚠️  Index creation failed for {column}: {e}")
        
        # Commit all changes
        conn.commit()
        
        # Generate summary statistics
        print(f"\n📈 DATABASE SUMMARY")
        print("=" * 70)
        
        # Basic stats
        cursor = conn.execute("SELECT COUNT(*) FROM relay_transactions")
        total_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT wallet) FROM relay_transactions")
        unique_wallets = cursor.fetchone()[0]
        
        # Date range
        cursor = conn.execute("""
            SELECT MIN(created_at) as first_tx, MAX(created_at) as last_tx 
            FROM relay_transactions
        """)
        date_range = cursor.fetchone()
        
        # Chain distribution
        cursor = conn.execute("""
            SELECT origin_chain_name, COUNT(*) as tx_count
            FROM relay_transactions 
            GROUP BY origin_chain_name 
            ORDER BY tx_count DESC
            LIMIT 5
        """)
        top_chains = cursor.fetchall()
        
        print(f"✅ Database created: {db_name}")
        print(f"📊 Total transactions: {total_count:,}")
        print(f"👤 Unique wallets: {unique_wallets:,}")
        print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
        print(f"\nTop origin chains:")
        for chain, count in top_chains:
            print(f"   {chain}: {count:,} transactions")
        
        print(f"\n🔍 Example queries:")
        print(f"   conn = sqlite3.connect('{db_name}')")
        print(f"   df = pd.read_sql('SELECT * FROM relay_transactions LIMIT 5', conn)")
        print(f"   # User segmentation: wallet transaction counts")
        print(f"   # Loyalty analysis: repeat users by time period")
        print(f"   # Chain preferences: origin->destination patterns")
        
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        
    finally:
        conn.close()
        print(f"\n🔒 Database connection closed")

def quick_test(db_name="relay_analysis.db"):
    """
    Quick verification of database structure and sample data.
    """
    try:
        conn = sqlite3.connect(db_name)
        
        # Test basic query
        df = pd.read_sql("""
            SELECT wallet, origin_chain_name, destination_chain_name, 
                   user_send_currency, user_receive_currency,
                   DATE(created_at) as date
            FROM relay_transactions 
            ORDER BY created_at 
            LIMIT 5
        """, conn)
        
        print("\n🧪 SAMPLE DATA (First 5 transactions):")
        print(df.to_string(index=False))
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Test query failed: {e}")
        return False

if __name__ == "__main__":
    # Create the database
    create_relay_database()
    
    # Quick verification
    quick_test()
