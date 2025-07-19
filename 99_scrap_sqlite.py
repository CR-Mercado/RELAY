# THIS FILE CAN BE IGNORED ONLY USED FOR RANDOM SQL QUERIES // CHECKING OTHER FILES SQL

import sqlite3
import pandas as pd

# Database connection
conn = sqlite3.connect("relay_analysis.db")

# PLEASE NOTE THESE COLUMN NAMES 
# created_at, wallet, execution_kind, is_call, route_source, deposit_tx_hash,
# origin_chain_name, user_send_currency, user_send_currency_amount, 
# user_send_currency_usd, destination_chain_name, 
# fill_tx_hash, user_receive_currency_amount, user_receive_currency_usd, user_receive_currency

#
# Put your SQL code here
sql = """
SELECT fill_tx_hash FROM relay_transactions 
WHERE wallet = '0xf909c4ae16622898b885b89d7f839e0244851c66'
and origin_chain_name = 'berachain'
and destination_chain_name = 'solana'
limit 2
"""

# Execute and show results
df = pd.read_sql(sql, conn)
print(df)

# Export to CSV
df.to_csv("99_output.csv", index=False)
print(f"\n💾 Results saved to: 99_output.csv")

# Close connection
conn.close() 