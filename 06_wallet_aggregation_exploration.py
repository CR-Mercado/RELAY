import sqlite3
import pandas as pd

def load_wallet_features(db_name="relay_analysis.db"):
    """
    Load wallet features from the SQLite database table.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        df = pd.read_sql("SELECT * FROM wallet_features_exclusions_applied", conn)
        print(f"✅ Loaded {len(df):,} wallets from database")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Make sure to run 05_exclusions_applied.py first")
        return None
    finally:
        conn.close()

def create_summary_stats_csv(df, output_file="wallet_features_summary_stats.csv"):
    """
    Create a CSV with summary statistics for each column.
    """
    
    numeric_cols = ['origin_chains', 'dest_chains', 'currency_sends', 'call_count', 'distinct_routes', 
                   'cross_chain_swaps', 'bridges', 'unique_days', 'deposit_tx_count', 'total_send_usd', 'total_receive_usd']
    
    # Create summary statistics dataframe
    summary_data = []
    
    for col in numeric_cols:
        stats = {
            'column': col,
            'count': df[col].count(),
            'mean': round(df[col].mean(), 2),
            'median': round(df[col].median(), 2),
            'q1': round(df[col].quantile(0.25), 2),
            'q3': round(df[col].quantile(0.75), 2),
            'iqr': round(df[col].quantile(0.75) - df[col].quantile(0.25), 2),
            'p90': round(df[col].quantile(0.90), 2),
            'p99': round(df[col].quantile(0.99), 2),
            'min': df[col].min(),
            'max': df[col].max(),
            'std_dev': round(df[col].std(), 2)
        }
        summary_data.append(stats)
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    
    print(f"✅ Summary statistics saved to: {output_file}")
    return summary_df

def print_summary_stats(df):
    """
    Print detailed summary statistics.
    """
    
    numeric_cols = ['origin_chains', 'dest_chains', 'currency_sends', 'call_count', 'distinct_routes', 
                   'cross_chain_swaps', 'bridges', 'unique_days', 'deposit_tx_count', 'total_send_usd', 'total_receive_usd']
    
    print("\n📈 WALLET FEATURE SUMMARY STATISTICS")
    print("=" * 80)
    
    for col in numeric_cols:
        print(f"\n{col.upper().replace('_', ' ')}:")
        print(f"   Count: {df[col].count():,}")
        print(f"   Mean: {df[col].mean():.2f}")
        print(f"   Median: {df[col].median():.2f}")
        print(f"   Q1 (25th): {df[col].quantile(0.25):.2f}")
        print(f"   Q3 (75th): {df[col].quantile(0.75):.2f}")
        print(f"   IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
        print(f"   P90 (90th): {df[col].quantile(0.90):.2f}")
        print(f"   P99 (99th): {df[col].quantile(0.99):.2f}")
        print(f"   Min: {df[col].min()}")
        print(f"   Max: {df[col].max()}")
        print(f"   Std Dev: {df[col].std():.2f}")

def create_html_table(summary_df, output_file="wallet_features_summary_table.html"):
    """
    Create an aesthetic HTML table from the summary statistics.
    """
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wallet Features Summary Statistics</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            .container {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 12px 8px;
                text-align: center;
                font-weight: bold;
                border: 1px solid #2980b9;
            }}
            td {{
                padding: 10px 8px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #e8f4f8;
            }}
            .column-name {{
                font-weight: bold;
                text-align: left;
                background-color: #ecf0f1;
                color: #2c3e50;
            }}
            .number {{
                font-family: 'Courier New', monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Wallet Features Summary Statistics</h1>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Count</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Q1</th>
                        <th>Q3</th>
                        <th>IQR</th>
                        <th>P90</th>
                        <th>P99</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Std Dev</th>
                    </tr>
                </thead>
                <tbody>
    {table_rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    # Generate table rows
    table_rows = ""
    for _, row in summary_df.iterrows():
        table_rows += f"""
                    <tr>
                        <td class="column-name">{row['column'].replace('_', ' ').title()}</td>
                        <td class="number">{row['count']:,}</td>
                        <td class="number">{row['mean']}</td>
                        <td class="number">{row['median']}</td>
                        <td class="number">{row['q1']}</td>
                        <td class="number">{row['q3']}</td>
                        <td class="number">{row['iqr']}</td>
                        <td class="number">{row['p90']}</td>
                        <td class="number">{row['p99']}</td>
                        <td class="number">{row['min']}</td>
                        <td class="number">{row['max']}</td>
                        <td class="number">{row['std_dev']}</td>
                    </tr>"""
    
    # Write HTML file
    html_content = html_template.format(table_rows=table_rows)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML table saved to: {output_file}")

if __name__ == "__main__":
    # Load data from database
    df = load_wallet_features()
    
    if df is not None and len(df) > 0:
        # Print summary statistics
        print_summary_stats(df)
        
        # Create summary statistics CSV
        print(f"\n📊 Creating summary statistics CSV...")
        summary_df = create_summary_stats_csv(df)
        
        # Create aesthetic HTML table
        print(f"\n🎨 Creating HTML table...")
        create_html_table(summary_df)
        
    else:
        print("❌ No data to analyze") 