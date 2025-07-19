import sqlite3
import pandas as pd
from pathlib import Path

def get_top_wallets(db_name="relay_analysis.db", limit=1000, export_csv=True):
    """
    Query top wallets by transaction count from Relay SQLite database.
    Similar to R's dbGetQuery() for wallet segmentation analysis.
    """
    
    # Check if database exists
    if not Path(db_name).exists():
        print(f"❌ Database not found: {db_name}")
        print("   Run 02_makesqlite.py first to create the database")
        return None
    
    print(f"🔍 Querying top {limit:,} wallets from {db_name}")
    print("=" * 60)
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_name)
        
        # Main query: top wallets by transaction count
        query = """
        SELECT 
            wallet, 
            COUNT(deposit_tx_hash) as n_tx
        FROM relay_transactions 
        GROUP BY wallet 
        ORDER BY n_tx DESC 
        LIMIT ?
        """
        
        print("📊 Executing query...")
        df = pd.read_sql(query, conn, params=[limit])
        
        # Display summary statistics
        print(f"✅ Found {len(df):,} wallets")
        print(f"\nTransaction count distribution:")
        print(f"   Max transactions: {df['n_tx'].max():,}")
        print(f"   Min transactions: {df['n_tx'].min():,}")
        print(f"   Mean transactions: {df['n_tx'].mean():.1f}")
        print(f"   Median transactions: {df['n_tx'].median():.1f}")
        
        # Show top 10 wallets
        print(f"\n🏆 TOP 10 WALLETS:")
        print(df.head(10).to_string(index=False))
        
        # User segmentation insights (based on your notes)
        print(f"\n📈 USER SEGMENTATION INSIGHTS:")
        
        # New users (≤5 transactions)
        new_users = len(df[df['n_tx'] <= 5])
        print(f"   New to Relay (≤5 tx): {new_users:,} wallets ({new_users/len(df)*100:.1f}%)")
        
        # Advanced users (>100 transactions)
        advanced_users = len(df[df['n_tx'] > 100])
        print(f"   Advanced users (>100 tx): {advanced_users:,} wallets ({advanced_users/len(df)*100:.1f}%)")
        
        # Potential programmatic users (>1000 transactions)
        programmatic_users = len(df[df['n_tx'] > 1000])
        print(f"   Potential programmatic (>1000 tx): {programmatic_users:,} wallets ({programmatic_users/len(df)*100:.1f}%)")
        
        # Export to CSV for further analysis
        if export_csv:
            output_file = f"top_{limit}_wallets.csv"
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results exported to: {output_file}")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return None

def get_wallet_details(wallet_address, db_name="relay_analysis.db"):
    """
    Get detailed transaction history for a specific wallet.
    Similar to R's filter() and summarize() functions.
    """
    
    if not Path(db_name).exists():
        print(f"❌ Database not found: {db_name}")
        return None
    
    try:
        conn = sqlite3.connect(db_name)
        
        # Detailed wallet analysis
        query = """
        SELECT 
            wallet,
            COUNT(*) as total_transactions,
            COUNT(DISTINCT origin_chain_name) as origin_chains_used,
            COUNT(DISTINCT destination_chain_name) as dest_chains_used,
            COUNT(DISTINCT user_send_currency) as send_currencies_used,
            COUNT(DISTINCT user_receive_currency) as receive_currencies_used,
            MIN(created_at) as first_transaction,
            MAX(created_at) as last_transaction,
            SUM(user_send_currency_usd) as total_volume_usd
        FROM relay_transactions 
        WHERE wallet = ?
        GROUP BY wallet
        """
        
        df = pd.read_sql(query, conn, params=[wallet_address])
        
        if len(df) == 0:
            print(f"❌ Wallet not found: {wallet_address}")
            return None
        
        print(f"👤 WALLET ANALYSIS: {wallet_address}")
        print("=" * 80)
        
        row = df.iloc[0]
        print(f"Total transactions: {row['total_transactions']:,}")
        print(f"Origin chains used: {row['origin_chains_used']}")
        print(f"Destination chains used: {row['dest_chains_used']}")
        print(f"Send currencies used: {row['send_currencies_used']}")
        print(f"Receive currencies used: {row['receive_currencies_used']}")
        print(f"First transaction: {row['first_transaction']}")
        print(f"Last transaction: {row['last_transaction']}")
        print(f"Total volume (USD): ${row['total_volume_usd']:,.2f}")
        
        # Get recent transactions
        recent_query = """
        SELECT created_at, origin_chain_name, destination_chain_name,
               user_send_currency, user_receive_currency, user_send_currency_usd
        FROM relay_transactions 
        WHERE wallet = ?
        ORDER BY created_at DESC
        LIMIT 5
        """
        
        recent_df = pd.read_sql(recent_query, conn, params=[wallet_address])
        print(f"\n📋 RECENT TRANSACTIONS (last 5):")
        print(recent_df.to_string(index=False))
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"❌ Wallet analysis failed: {e}")
        return None

def analyze_user_segments(db_name="relay_analysis.db"):
    """
    Analyze user segments based on transaction patterns.
    Implements segmentation logic from your notes.
    """
    
    if not Path(db_name).exists():
        print(f"❌ Database not found: {db_name}")
        return None
    
    try:
        conn = sqlite3.connect(db_name)
        
        # User segmentation query
        query = """
        SELECT 
            wallet,
            COUNT(*) as tx_count,
            COUNT(DISTINCT origin_chain_name) as chains_used,
            COUNT(DISTINCT user_send_currency) as currencies_used,
            MIN(created_at) as first_tx,
            MAX(created_at) as last_tx,
            AVG(user_send_currency_usd) as avg_tx_size,
            CASE 
                WHEN COUNT(*) <= 5 THEN 'New to Relay'
                WHEN COUNT(*) > 1000 THEN 'Advanced Programmatic'
                WHEN COUNT(DISTINCT origin_chain_name) <= 2 THEN 'Beginner Explorer'
                ELSE 'Regular User'
            END as user_segment
        FROM relay_transactions 
        GROUP BY wallet
        """
        
        print("🎯 ANALYZING USER SEGMENTS...")
        df = pd.read_sql(query, conn)
        
        # Segment summary
        segment_summary = df['user_segment'].value_counts()
        
        print(f"\n📊 USER SEGMENT DISTRIBUTION:")
        for segment, count in segment_summary.items():
            percentage = count / len(df) * 100
            print(f"   {segment}: {count:,} users ({percentage:.1f}%)")
        
        # Export segments
        df.to_csv("user_segments.csv", index=False)
        print(f"\n💾 User segments exported to: user_segments.csv")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"❌ Segmentation analysis failed: {e}")
        return None

if __name__ == "__main__":
    # Get top 1000 wallets
    top_wallets = get_top_wallets(limit=1000)
    
    if top_wallets is not None:
        print(f"\n" + "="*60)
        
        # Analyze user segments
        segments = analyze_user_segments()
        
        # Show example of how to analyze a specific wallet
        if len(top_wallets) > 0:
            top_wallet = top_wallets.iloc[0]['wallet']
            print(f"\n" + "="*60)
            print(f"📋 Example: Analyzing top wallet...")
            get_wallet_details(top_wallet) 