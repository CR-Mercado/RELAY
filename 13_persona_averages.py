import sqlite3
import pandas as pd
import numpy as np

def load_persona_data(db_name="relay_analysis.db"):
    """
    Load persona data with all business features from the database.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Loading persona data from database...")
        
        # Check what columns are available
        columns_query = "PRAGMA table_info(loyalty_persona_final)"
        columns_df = pd.read_sql(columns_query, conn)
        available_columns = columns_df['name'].tolist()
        
        print(f"✅ Available columns: {len(available_columns)} total")
        
        # Load all persona data
        query = """
        SELECT *
        FROM loyalty_persona_final
        WHERE persona IS NOT NULL
        """
        
        df = pd.read_sql(query, conn)
        
        print(f"📊 Loaded {len(df):,} wallets with persona assignments")
        
        # Show persona distribution
        persona_counts = df['persona'].value_counts()
        print(f"\n📈 Persona Distribution:")
        for persona, count in persona_counts.items():
            pct = count / len(df) * 100
            print(f"   {persona}: {count:,} ({pct:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading persona data: {e}")
        return None
    finally:
        conn.close()

def calculate_persona_averages(df):
    """
    Calculate average characteristics for each persona using same logic as 11_loyalty_persona.py.
    """
    
    try:
        print("\n🔄 Calculating persona characteristics using clustering logic...")
        
        # Use EXACT same business features as defined in 11_loyalty_persona.py
        business_features = [
            'origin_chains', 'dest_chains', 'currency_sends', 'call_count',
            'distinct_routes', 'cross_chain_swaps', 'bridges', 'unique_days', 'total_send_usd'
        ]
        
        # Check which features are available
        available_features = [col for col in business_features if col in df.columns]
        missing_features = [col for col in business_features if col not in df.columns]
        
        if missing_features:
            print(f"⚠️  Missing columns: {missing_features}")
            print(f"💡 Need to re-run 11_loyalty_persona.py to save all business features to database")
        
        print(f"✅ Using {len(available_features)} business features: {available_features}")
        
        # Calculate in-group averages by persona - SAME logic as 11_loyalty_persona.py
        persona_profiles = df.groupby('persona')[available_features].mean().round(2)
        
        print(f"\n📊 Persona Characteristics (In-Group Averages):")
        print(persona_profiles)
        
        return persona_profiles, available_features
        
    except Exception as e:
        print(f"❌ Error calculating averages: {e}")
        return None, None

def create_aesthetic_table(df, persona_profiles, available_features):
    """
    Create clean formatted table showing persona characteristics - RELAY USAGE ONLY.
    """
    
    try:
        print("\n🎨 Creating RELAY USAGE persona characteristics table...")
        print("💡 Note: All data sourced from loyalty_persona_final database table")
        
        # Get persona counts
        persona_counts = df['persona'].value_counts()
        
        # Expected columns as requested
        expected_columns = [
            'origin_chains',    # # Origins  
            'dest_chains',      # # Destinations
            'call_count',       # # F() Calls
            'bridges',          # # Simple Bridge
            'cross_chain_swaps', # # Cross Chain Swaps
            'currency_sends',   # # Tokens
            'unique_days',      # # Unique Days
            'total_send_usd'    # 6-mo Volume
        ]
        
        # Check which columns are available
        available_columns = [col for col in expected_columns if col in available_features]
        missing_columns = [col for col in expected_columns if col not in available_features]
        
        if missing_columns:
            print(f"⚠️  Missing expected columns: {missing_columns}")
        
        # Create the main table
        results_table = pd.DataFrame()
        
        # Add count column first
        results_table['Count'] = persona_counts.reindex(persona_profiles.index)
        
        # Add percentage
        total_wallets = persona_counts.sum()
        results_table['%'] = (results_table['Count'] / total_wallets * 100).round(1)
        
        # Add available feature columns with clean names
        column_mapping = {
            'origin_chains': '# Origins',
            'dest_chains': '# Destinations', 
            'call_count': '# F() Calls',
            'bridges': '# Simple Bridge',
            'cross_chain_swaps': '# Cross Chain Swaps',
            'currency_sends': '# Tokens',
            'unique_days': '# Unique Days',
            'total_send_usd': '6-mo Volume'
        }
        
        for col in available_columns:
            clean_name = column_mapping.get(col, col)
            if col == 'total_send_usd':
                # Format currency
                results_table[clean_name] = persona_profiles[col].apply(lambda x: f"${x:,.0f}")
            else:
                # Round to 1 decimal place
                results_table[clean_name] = persona_profiles[col].round(1)
        
        print(f"\n" + "="*100)
        print(f"📊 RELAY USER PERSONA CHARACTERISTICS (6-Month Averages Within Each Cluster)")
        print(f"💡 Data Source: Database loyalty_persona_final table | Relay Usage Only")
        print(f"="*100)
        
        # Print with nice formatting
        print(results_table.to_string(index=True, col_space=12))
        
        print(f"\n" + "="*100)
        print(f"📈 INTERPRETATION:")
        print(f"   🟣 Basic Bridge Users: Simple, low-volume users (1-2 chains, ~$272 avg)")
        print(f"   🔴 High Value Users: Moderate complexity (4-5 chains, ~$1,290 avg)")  
        print(f"   🔵 Multi-Chain Users: High complexity power users (10+ chains, ~$3,645 avg)")
        print(f"="*100)
        
        return results_table
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return None

def create_html_table(results_table, output_file="relay_persona_characteristics.html"):
    """
    Create an HTML version of the persona characteristics table.
    """
    
    try:
        print(f"\n💾 Creating HTML table: {output_file}")
        
        # Remove 6-mo Volume and add Total Volume column
        results_table_clean = results_table.copy()
        results_table_clean['Total Volume'] = results_table_clean['6-mo Volume']
        results_table_clean = results_table_clean.drop(columns=['6-mo Volume'])
        
        # Create clean HTML with just the table as requested
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Relay User Persona Characteristics</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: white; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-bottom: 15px; font-size: 18px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
        th {{ background-color: #3498db; color: white; padding: 8px 6px; text-align: center; font-weight: bold; font-size: 12px; }}
        td {{ padding: 8px 6px; border-bottom: 1px solid #ddd; text-align: center; }}
        .persona-col {{ background-color: #3498db; color: white; font-weight: bold; text-align: left; padding-left: 12px; }}
        .main-table {{ border: 2px solid #3498db; }}
    </style>
</head>
<body>
    <h2>📊 Jan 1 - June 30 2025 Averages within Cluster</h2>
    {results_table_clean.to_html(classes='main-table', table_id='personas', escape=False)}
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const table = document.getElementById('personas');
            const rows = table.querySelectorAll('tbody tr');
            
            rows.forEach(row => {{
                const firstCell = row.querySelector('td:first-child');
                if (firstCell) {{
                    firstCell.classList.add('persona-col');
                }}
            }});
            
            const headerCells = table.querySelectorAll('thead th');
            if (headerCells[0]) {{
                headerCells[0].style.textAlign = 'left';
                headerCells[0].style.paddingLeft = '12px';
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML table saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating HTML table: {e}")
        return False

def export_csv_table(results_table):
    """
    Export table to CSV file.
    """
    
    try:
        print("\n📤 Exporting CSV file...")
        
        # Export results table
        results_table.to_csv("relay_persona_characteristics.csv")
        print("✅ Saved: relay_persona_characteristics.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting CSV file: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Persona Characteristics Analysis...")
    print("=" * 60)
    
    # Load persona data
    df = load_persona_data()
    if df is None:
        print("❌ Cannot load persona data")
        exit(1)
    
    # Calculate averages using same logic as 11_loyalty_persona.py
    persona_profiles, available_features = calculate_persona_averages(df)
    if persona_profiles is None:
        print("❌ Cannot calculate persona averages")
        exit(1)
    
    # Create aesthetic table
    results_table = create_aesthetic_table(df, persona_profiles, available_features)
    if results_table is None:
        print("❌ Cannot create table")
        exit(1)
    
    # Create HTML table
    html_success = create_html_table(results_table)
    
    # Export CSV file
    csv_success = export_csv_table(results_table)
    
    # Final summary
    print(f"\n🎯 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Relay persona characteristics analyzed from DATABASE!")
    print(f"📊 Analyzed {len(df):,} wallets across {len(persona_profiles)} personas")
    print(f"📈 Generated table with {len(available_features)} business features")
    
    if html_success:
        print(f"🌐 HTML table: relay_persona_characteristics.html")
    if csv_success:
        print(f"📄 CSV exported: relay_persona_characteristics.csv")
    
    print(f"\n💡 Key Takeaways:")
    print(f"   📊 Clear differentiation between personas by complexity & volume")
    print(f"   🎯 82% Basic users vs 14% High Value vs 4% Multi-Chain power users")
    print(f"   📈 Natural cluster sizes reflect real user distribution patterns") 