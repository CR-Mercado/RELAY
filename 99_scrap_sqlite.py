# THIS FILE CAN BE IGNORED ONLY USED FOR RANDOM SQL QUERIES // CHECKING OTHER FILES SQL

import sqlite3
import pandas as pd

# Database connection
conn = sqlite3.connect("relay_analysis.db")

# Put your SQL code here
sql = """
SELECT strftime('%Y-%m', created_at) as month_,
COUNT(deposit_tx_hash) as n_tx 
FROM relay_transactions 
GROUP BY month_
ORDER BY month_ DESC
"""

# Execute and show results
df = pd.read_sql(sql, conn)
print(df)

# Close connection
conn.close() 