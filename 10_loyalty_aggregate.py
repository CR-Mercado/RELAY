import sqlite3
import pandas as pd

def create_loyalty_aggregate_table(db_name="relay_analysis.db", table_name="loyalty_aggregate"):
    """
    Create a table combining Relay users with their competitive platform usage.
    This enables analysis of user loyalty and cross-platform behavior.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Creating loyalty aggregate table...")
        
        # Create the loyalty aggregate table
        query = f"""
        DROP TABLE IF EXISTS {table_name};
        
        CREATE TABLE {table_name} AS
        WITH exclusions AS (
            SELECT '0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae' as wallet
            UNION ALL SELECT '0x4f8c9056bb8a3616693a76922fa35d53c056e5b3'
            UNION ALL SELECT '0xde1e598b81620773454588b85d6b5d4eec32573e'
            UNION ALL SELECT '0x864b314d4c5a0399368609581d3e8933a63b9232'
            UNION ALL SELECT '0x341e94069f53234fe6dabef707ad424830525715'
            UNION ALL SELECT '0x896ff3b31ecc105d4f23582c73416484ecc207c6'
            UNION ALL SELECT '0xf909c4ae16622898b885b89d7f839e0244851c66'
            UNION ALL SELECT '72z3PVWqpzyDd4CQGsQJU6eGt6fLq4D3ULrSKdMULyY8'
            UNION ALL SELECT 'zApVWDs3nSychNnUXSS2czhY78Ycopa15zELrK2gAdM'
            UNION ALL SELECT '5trW3ZogRMxW9tX4pNCMQi1APzx8G6UTcyneWEX2Rk4Q'
            UNION ALL SELECT '9s3Kyeg2NHeM2xhSqMyp944f8VS2oBYdwvK98AAnW35w'
            UNION ALL SELECT '0x0000000000000000000000000000000000000000'
            UNION ALL SELECT '0x000000000000000000000000000000000000dead'
        ),
        relay_aggregated AS (
            SELECT 
                wallet as wallet_,
                origin_chain_name as source_chain,
                'Relay' as platform_,
                destination_chain_name as destination_chain,
                SUM(user_send_currency_usd) as amount_usd,
                COUNT(*) as tx_count
            FROM relay_transactions
            WHERE wallet_ NOT IN (SELECT wallet FROM exclusions)
                AND created_at >= '2025-01-01'
            GROUP BY wallet_, origin_chain_name, destination_chain_name
        ),
        competitive_data AS (
            SELECT 
                wallet_,
                source_chain,
                platform_,
                destination_chain,
                amount_usd,
                tx_count
            FROM across_debridge_usage 
            WHERE wallet_ NOT IN (SELECT wallet FROM exclusions)
                AND wallet_ IN (SELECT wallet_ FROM relay_aggregated)
        )
        SELECT 
            wallet_,
            source_chain,
            platform_,
            destination_chain,
            amount_usd,
            tx_count
        FROM relay_aggregated
        UNION ALL
        SELECT * FROM competitive_data;
        """
        
        conn.executescript(query)
        
        # Create indexes for performance
        conn.execute(f"CREATE INDEX idx_{table_name}_platform ON {table_name}(platform_);")
        conn.execute(f"CREATE INDEX idx_{table_name}_route ON {table_name}(source_chain, destination_chain);")
        
        # Get summary statistics
        summary_query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT wallet_) as unique_wallets,
            COUNT(DISTINCT platform_) as unique_platforms,
            COUNT(DISTINCT source_chain) as unique_source_chains,
            COUNT(DISTINCT destination_chain) as unique_dest_chains,
            SUM(amount_usd) as total_volume_usd,
            SUM(tx_count) as total_transactions
        FROM {table_name}
        """
        
        summary_df = pd.read_sql(summary_query, conn)
        
        print("✅ Loyalty aggregate table created successfully!")
        print("\n📊 SUMMARY STATISTICS:")
        print(f"   Total records: {summary_df['total_records'].iloc[0]:,}")
        print(f"   Unique wallets: {summary_df['unique_wallets'].iloc[0]:,}")
        print(f"   Unique platforms: {summary_df['unique_platforms'].iloc[0]}")
        print(f"   Unique source chains: {summary_df['unique_source_chains'].iloc[0]}")
        print(f"   Unique destination chains: {summary_df['unique_dest_chains'].iloc[0]}")
        print(f"   Total volume: ${summary_df['total_volume_usd'].iloc[0]:,.2f}")
        print(f"   Total transactions: {summary_df['total_transactions'].iloc[0]:,}")
        
    except Exception as e:
        print(f"❌ Error creating loyalty aggregate table: {e}")
        return False
    finally:
        conn.close()
    
    return True

def analyze_platform_chains(db_name="relay_analysis.db", table_name="loyalty_aggregate"):
    """
    Analyze distinct platform/chain combinations to identify naming inconsistencies.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print(f"\n🔍 PLATFORM/CHAIN ANALYSIS")
        print("=" * 60)
        
        # Platform source chain combinations
        print(f"\n🌉 Platform + Source Chain Combinations:")
        platform_source_query = f"""
        SELECT 
            platform_,
            source_chain,
            COUNT(DISTINCT wallet_) as unique_wallets,
            SUM(amount_usd) as total_volume,
            SUM(tx_count) as total_transactions
        FROM {table_name}
        GROUP BY platform_, source_chain
        ORDER BY platform_, total_volume DESC
        """
        
        platform_source_df = pd.read_sql(platform_source_query, conn)
        print(platform_source_df.to_string(index=False))
        
        # Platform destination chain combinations
        print(f"\n🎯 Platform + Destination Chain Combinations:")
        platform_dest_query = f"""
        SELECT 
            platform_,
            destination_chain,
            COUNT(DISTINCT wallet_) as unique_wallets,
            SUM(amount_usd) as total_volume,
            SUM(tx_count) as total_transactions
        FROM {table_name}
        GROUP BY platform_, destination_chain
        ORDER BY platform_, total_volume DESC
        """
        
        platform_dest_df = pd.read_sql(platform_dest_query, conn)
        print(platform_dest_df.to_string(index=False))
        
        # Unique chain names across all platforms (for consistency checking)
        print(f"\n🔗 All Unique Chain Names (Source):")
        unique_source_query = f"""
        SELECT DISTINCT source_chain, COUNT(DISTINCT platform_) as platforms_count
        FROM {table_name}
        GROUP BY source_chain
        ORDER BY source_chain
        """
        
        unique_source_df = pd.read_sql(unique_source_query, conn)
        print(unique_source_df.to_string(index=False))
        
        print(f"\n🔗 All Unique Chain Names (Destination):")
        unique_dest_query = f"""
        SELECT DISTINCT destination_chain, COUNT(DISTINCT platform_) as platforms_count
        FROM {table_name}
        GROUP BY destination_chain
        ORDER BY destination_chain
        """
        
        unique_dest_df = pd.read_sql(unique_dest_query, conn)
        print(unique_dest_df.to_string(index=False))
        
        # Save analysis files
        platform_source_df.to_csv("platform_source_combinations.csv", index=False)
        platform_dest_df.to_csv("platform_dest_combinations.csv", index=False)
        unique_source_df.to_csv("unique_source_chains.csv", index=False)
        unique_dest_df.to_csv("unique_dest_chains.csv", index=False)
        
        print(f"\n💾 Analysis saved to CSV files:")
        print(f"   - platform_source_combinations.csv")
        print(f"   - platform_dest_combinations.csv") 
        print(f"   - unique_source_chains.csv")
        print(f"   - unique_dest_chains.csv")
        
        # Highlight potential naming inconsistencies
        print(f"\n⚠️  CHAIN NAME CONSISTENCY CHECK:")
        all_chains = set(unique_source_df['source_chain'].tolist() + unique_dest_df['destination_chain'].tolist())
        print(f"   Total unique chain names: {len(all_chains)}")
        print(f"   Chain names to review for consistency:")
        for chain in sorted(all_chains):
            print(f"     - {chain}")
        
    except Exception as e:
        print(f"❌ Error analyzing platform chains: {e}")
        return False
    finally:
        conn.close()
    
    return True

def preview_loyalty_data(db_name="relay_analysis.db", table_name="loyalty_aggregate", n_rows=10):
    """
    Preview the loyalty aggregate data.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print(f"\n👀 LOYALTY AGGREGATE DATA PREVIEW ({n_rows} rows)")
        print("=" * 80)
        
        preview_query = f"""
        SELECT *
        FROM {table_name}
        ORDER BY wallet_, platform_, amount_usd DESC
        LIMIT {n_rows}
        """
        
        preview_df = pd.read_sql(preview_query, conn)
        print(preview_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error previewing data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 Starting loyalty aggregate analysis...")
    
    # Create the loyalty aggregate table
    success = create_loyalty_aggregate_table()
    
    if success:
        # Analyze platform/chain combinations
        analyze_platform_chains()
        
        # Preview the data
        preview_loyalty_data()
        
        print(f"\n🎯 SUMMARY")
        print("=" * 60)
        print(f"✅ Loyalty aggregate table created: loyalty_aggregate")
        print(f"📊 Platform/chain analysis completed")
        print(f"📈 CSV files generated for chain name review")
        print(f"\n💡 Next steps:")
        print(f"   1. Review CSV files for chain naming inconsistencies")
        print(f"   2. Add CASE WHEN logic to normalize chain names")
        print(f"   3. Analyze user loyalty patterns across platforms")
        print(f"   4. Calculate Relay market share per user/route")
        
    else:
        print("❌ Failed to create loyalty aggregate table") 