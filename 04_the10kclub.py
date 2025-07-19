import sqlite3
import pandas as pd

def analyze_10k_club(db_name="relay_analysis.db"):
    """
    Analyze wallets with >10k transactions and their chain preferences.
    """
    
    conn = sqlite3.connect(db_name)
    
    # Get wallets with >10k transactions and their top 3 chains
    sql = """
    WITH wallet_counts AS (
        SELECT wallet, COUNT(*) as total_tx
        FROM relay_transactions 
        GROUP BY wallet 
        HAVING COUNT(*) > 10000
    ),
    
    wallet_origin_chains AS (
        SELECT 
            rt.wallet,
            rt.origin_chain_name,
            COUNT(*) as origin_count,
            ROW_NUMBER() OVER (PARTITION BY rt.wallet ORDER BY COUNT(*) DESC) as origin_rank
        FROM relay_transactions rt
        INNER JOIN wallet_counts wc ON rt.wallet = wc.wallet
        GROUP BY rt.wallet, rt.origin_chain_name
    ),
    
    wallet_dest_chains AS (
        SELECT 
            rt.wallet,
            rt.destination_chain_name,
            COUNT(*) as dest_count,
            ROW_NUMBER() OVER (PARTITION BY rt.wallet ORDER BY COUNT(*) DESC) as dest_rank
        FROM relay_transactions rt
        INNER JOIN wallet_counts wc ON rt.wallet = wc.wallet
        GROUP BY rt.wallet, rt.destination_chain_name
    ),
    
    ranks AS (
        SELECT 1 as rank UNION SELECT 2 UNION SELECT 3
    )
    
    SELECT 
        wc.wallet,
        wc.total_tx,
        r.rank as source_rank,
        woc.origin_chain_name as source,
        woc.origin_count as source_count,
        r.rank as dest_rank,
        wdc.destination_chain_name as dest,
        wdc.dest_count as dest_count
    FROM wallet_counts wc
    CROSS JOIN ranks r
    LEFT JOIN wallet_origin_chains woc ON wc.wallet = woc.wallet AND woc.origin_rank = r.rank
    LEFT JOIN wallet_dest_chains wdc ON wc.wallet = wdc.wallet AND wdc.dest_rank = r.rank
    ORDER BY wc.total_tx DESC, r.rank
    """
    
    df = pd.read_sql(sql, conn)
    
    print("🏆 THE 10K CLUB - Wallets with >10,000 transactions")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Export results
    df.to_csv("10k_club_analysis.csv", index=False)
    print(f"\n💾 Results saved to: 10k_club_analysis.csv")
    
    conn.close()
    return df

if __name__ == "__main__":
    analyze_10k_club() 