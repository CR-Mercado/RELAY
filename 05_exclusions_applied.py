import sqlite3

def create_wallet_features_table(db_name="relay_analysis.db"):
    """
    Create wallet_features_exclusions_applied table with cleaned user data.
    """
    
    conn = sqlite3.connect(db_name)
    
    # SQL with CREATE TABLE and INSERT statements
    sql = """
    -- Drop existing table if it exists
    DROP TABLE IF EXISTS wallet_features_exclusions_applied;
    
    -- Create new table and populate it
    CREATE TABLE wallet_features_exclusions_applied AS
    WITH exclusions_ AS (
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
    
    wallet_stats AS (
        SELECT 
            rt.wallet,
            COUNT(DISTINCT rt.origin_chain_name) as origin_chains,
            COUNT(DISTINCT rt.destination_chain_name) as dest_chains,
            COUNT(DISTINCT rt.user_send_currency) as currency_sends,
            SUM(CASE WHEN rt.is_call = 1 THEN 1 ELSE 0 END) as call_count,
            COUNT(DISTINCT rt.route_source) as distinct_routes,
            SUM(CASE WHEN rt.execution_kind = 'cross-chain-swap' THEN 1 ELSE 0 END) as cross_chain_swaps,
            SUM(CASE WHEN rt.execution_kind = 'bridge' THEN 1 ELSE 0 END) as bridges,
            COUNT(DISTINCT DATE(rt.created_at)) as unique_days,
            COUNT(rt.deposit_tx_hash) as deposit_tx_count,
            SUM(rt.user_send_currency_usd) as total_send_usd,
            SUM(rt.user_receive_currency_usd) as total_receive_usd
        FROM relay_transactions rt
        WHERE NOT EXISTS (
            SELECT 1 FROM exclusions_ ex WHERE ex.wallet = rt.wallet
        )
        GROUP BY rt.wallet
    )
    
    SELECT 
        wallet,
        origin_chains,
        dest_chains, 
        currency_sends,
        call_count,
        distinct_routes,
        cross_chain_swaps,
        bridges,
        unique_days,
        deposit_tx_count,
        total_send_usd,
        total_receive_usd
    FROM wallet_stats;
    """
    
    print("🔄 Creating wallet_features_exclusions_applied table...")
    
    # Execute the SQL statements
    conn.executescript(sql)
    conn.close()
    
    print("✅ Table created successfully")

if __name__ == "__main__":
    create_wallet_features_table()
