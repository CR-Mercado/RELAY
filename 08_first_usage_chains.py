import sqlite3
import pandas as pd

def create_first_usage_chains_table(db_name="relay_analysis.db"):
    """
    Create a table with each wallet's first source and destination chains.
    Excludes known programmatic/burn addresses.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Creating first usage chains analysis...")
        
        # Create the first usage chains table
        query = """
        DROP TABLE IF EXISTS wallet_first_usage_chains;
        
        CREATE TABLE wallet_first_usage_chains AS
        WITH exclusions AS (
            SELECT '0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae' as wallet, 'LiFi-Diamond' as reasoning
            UNION ALL SELECT '0x4f8c9056bb8a3616693a76922fa35d53c056e5b3', 'LiFi-Diamond'
            UNION ALL SELECT '0xde1e598b81620773454588b85d6b5d4eec32573e', 'LiFi-Diamond'
            UNION ALL SELECT '0x864b314d4c5a0399368609581d3e8933a63b9232', 'LiFi-Diamond'
            UNION ALL SELECT '0x341e94069f53234fe6dabef707ad424830525715', 'LiFi-Diamond'
            UNION ALL SELECT '0x896ff3b31ecc105d4f23582c73416484ecc207c6', 'LiFi-Diamond'
            UNION ALL SELECT '0xf909c4ae16622898b885b89d7f839e0244851c66', 'LiFi-Diamond'
            UNION ALL SELECT '72z3PVWqpzyDd4CQGsQJU6eGt6fLq4D3ULrSKdMULyY8', 'thaolunar.sol'
            UNION ALL SELECT 'zApVWDs3nSychNnUXSS2czhY78Ycopa15zELrK2gAdM', 'thaolunar.sol'
            UNION ALL SELECT '5trW3ZogRMxW9tX4pNCMQi1APzx8G6UTcyneWEX2Rk4Q', 'thaolunar.sol'
            UNION ALL SELECT '9s3Kyeg2NHeM2xhSqMyp944f8VS2oBYdwvK98AAnW35w', 'thaolunar.sol'
            UNION ALL SELECT '0x0000000000000000000000000000000000000000', 'burn-address'
            UNION ALL SELECT '0x000000000000000000000000000000000000dead', 'burn-address'
        ),
                 wallet_first_tx AS (
             SELECT 
                 wallet,
                 MIN(created_at) as first_tx_time
             FROM relay_transactions
             WHERE wallet NOT IN (SELECT wallet FROM exclusions)
                 AND created_at >= '2025-01-01'  -- Only 2025 data (6 month period)
             GROUP BY wallet
         ),
         first_usage_details AS (
             SELECT 
                 rt.wallet,
                 rt.created_at as first_tx_time,
                 rt.origin_chain_name as first_source_chain,
                 rt.destination_chain_name as first_dest_chain,
                 rt.user_send_currency as first_currency,
                 rt.user_send_currency_usd as first_amount_usd,
                 rt.is_call as first_is_call
             FROM relay_transactions rt
                         INNER JOIN wallet_first_tx wft 
                 ON rt.wallet = wft.wallet 
                 AND rt.created_at = wft.first_tx_time
             WHERE rt.created_at >= '2025-01-01'
        )
        SELECT 
            wallet,
            first_tx_time,
            first_source_chain,
            first_dest_chain,
            first_currency,
            first_amount_usd,
            first_is_call,
            -- Add some derived flags for analysis
            CASE 
                WHEN first_source_chain = first_dest_chain THEN 1 
                ELSE 0 
            END as first_tx_same_chain,
            CASE 
                WHEN first_is_call = 1 THEN 'Call' 
                ELSE 'Not Call' 
            END as first_tx_type
        FROM first_usage_details
        ORDER BY first_tx_time;
        """
        
        conn.executescript(query)
        
        # Create index for faster joins
        conn.execute("CREATE INDEX idx_wallet_first_usage ON wallet_first_usage_chains(wallet);")
        
        # Get summary statistics
        summary_query = """
        SELECT 
            COUNT(*) as total_wallets,
            COUNT(DISTINCT first_source_chain) as unique_source_chains,
            COUNT(DISTINCT first_dest_chain) as unique_dest_chains,
            AVG(first_amount_usd) as avg_first_amount,
            SUM(first_tx_same_chain) as same_chain_first_txs,
            SUM(CASE WHEN first_is_call = 1 THEN 1 ELSE 0 END) as call_first_txs
        FROM wallet_first_usage_chains;
        """
        
        summary_df = pd.read_sql(summary_query, conn)
        
        print("✅ First usage chains table created successfully!")
        print("\n📊 SUMMARY STATISTICS:")
        print(f"   Total wallets analyzed: {summary_df['total_wallets'].iloc[0]:,}")
        print(f"   Unique first source chains: {summary_df['unique_source_chains'].iloc[0]}")
        print(f"   Unique first destination chains: {summary_df['unique_dest_chains'].iloc[0]}")
        print(f"   Average first transaction amount: ${summary_df['avg_first_amount'].iloc[0]:.2f}")
        print(f"   Same-chain first transactions: {summary_df['same_chain_first_txs'].iloc[0]:,}")
        print(f"   Call-type first transactions: {summary_df['call_first_txs'].iloc[0]:,}")
        
    except Exception as e:
        print(f"❌ Error creating first usage chains table: {e}")
        return False
    finally:
        conn.close()
    
    return True

def analyze_first_usage_patterns(db_name="relay_analysis.db"):
    """
    Analyze patterns in first usage chains.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print(f"\n🔍 FIRST USAGE PATTERNS ANALYSIS")
        print("=" * 60)
        
        # Top source chains for first transactions
        print("\n📈 Top 10 First Source Chains:")
        source_query = """
        SELECT 
            first_source_chain,
            COUNT(*) as wallet_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM wallet_first_usage_chains), 2) as percentage,
            ROUND(AVG(first_amount_usd), 2) as avg_amount
        FROM wallet_first_usage_chains
        GROUP BY first_source_chain
        ORDER BY wallet_count DESC
        LIMIT 10;
        """
        
        source_df = pd.read_sql(source_query, conn)
        print(source_df.to_string(index=False))
        
        # Top destination chains for first transactions
        print("\n📈 Top 10 First Destination Chains:")
        dest_query = """
        SELECT 
            first_dest_chain,
            COUNT(*) as wallet_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM wallet_first_usage_chains), 2) as percentage,
            ROUND(AVG(first_amount_usd), 2) as avg_amount
        FROM wallet_first_usage_chains
        GROUP BY first_dest_chain
        ORDER BY wallet_count DESC
        LIMIT 10;
        """
        
        dest_df = pd.read_sql(dest_query, conn)
        print(dest_df.to_string(index=False))
        
        # Top source -> destination pairs
        print("\n📈 Top 10 First Route Pairs (Source → Destination):")
        pairs_query = """
        SELECT 
            first_source_chain,
            first_dest_chain,
            COUNT(*) as wallet_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM wallet_first_usage_chains), 2) as percentage,
            ROUND(AVG(first_amount_usd), 2) as avg_amount
        FROM wallet_first_usage_chains
        GROUP BY first_source_chain, first_dest_chain
        ORDER BY wallet_count DESC
        LIMIT 10;
        """
        
        pairs_df = pd.read_sql(pairs_query, conn)
        print(pairs_df.to_string(index=False))
        
        # First transaction characteristics
        print("\n📊 First Transaction Characteristics:")
        chars_query = """
        SELECT 
            first_tx_type,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM wallet_first_usage_chains), 2) as percentage,
            ROUND(AVG(first_amount_usd), 2) as avg_amount,
            MIN(first_amount_usd) as min_amount,
            MAX(first_amount_usd) as max_amount
        FROM wallet_first_usage_chains
        GROUP BY first_tx_type;
        """
        
        chars_df = pd.read_sql(chars_query, conn)
        print(chars_df.to_string(index=False))
        
        # Save detailed analysis
        source_df.to_csv("first_usage_source_chains.csv", index=False)
        dest_df.to_csv("first_usage_dest_chains.csv", index=False)  
        pairs_df.to_csv("first_usage_route_pairs.csv", index=False)
        
        print(f"\n💾 Analysis saved to CSV files:")
        print(f"   - first_usage_source_chains.csv")
        print(f"   - first_usage_dest_chains.csv")
        print(f"   - first_usage_route_pairs.csv")
        
    except Exception as e:
        print(f"❌ Error analyzing first usage patterns: {e}")
        return False
    finally:
        conn.close()
    
    return True

def preview_first_usage_data(db_name="relay_analysis.db", n_rows=10):
    """
    Preview the first usage chains data.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print(f"\n👀 PREVIEW: First {n_rows} rows of wallet_first_usage_chains")
        print("=" * 80)
        
        preview_query = f"""
        SELECT *
        FROM wallet_first_usage_chains
        ORDER BY first_tx_time
        LIMIT {n_rows};
        """
        
        preview_df = pd.read_sql(preview_query, conn)
        print(preview_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error previewing data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 Starting first usage chains analysis...")
    
    # Create the first usage chains table
    success = create_first_usage_chains_table()
    
    if success:
        # Analyze patterns
        analyze_first_usage_patterns()
        
        # Preview the data
        preview_first_usage_data()
        
        print(f"\n🎯 SUMMARY")
        print("=" * 60)
        print(f"✅ First usage chains analysis completed")
        print(f"📊 Table created: wallet_first_usage_chains")
        print(f"📈 Analysis files: first_usage_*.csv")
        print(f"\n💡 Next steps:")
        print(f"   1. Join with wallet_features_exclusions_applied for segmentation analysis")
        print(f"   2. Analyze onboarding patterns by user volume/behavior")
        print(f"   3. Identify acquisition channels and chain preferences")
        
    else:
        print("❌ Failed to create first usage chains table") 