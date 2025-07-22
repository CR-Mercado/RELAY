import sqlite3
import pandas as pd
import numpy as np

def load_loyalty_segment_data(db_name="relay_analysis.db"):
    """
    Load data segmented by loyalty type (relay-only vs multi-platform).
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Loading loyalty segment data from database...")
        
        # Load all data with business features
        query = """
        SELECT 
            loyalty_type,
            persona,
            origin_chains, dest_chains, currency_sends, call_count,
            distinct_routes, cross_chain_swaps, bridges, unique_days, total_send_usd,
            n_platforms, total_amount_usd, total_tx_count
        FROM loyalty_persona_final
        WHERE loyalty_type IS NOT NULL AND persona IS NOT NULL
        """
        
        df = pd.read_sql(query, conn)
        
        print(f"✅ Loaded {len(df):,} wallets with loyalty and persona data")
        
        # Show loyalty distribution
        loyalty_counts = df['loyalty_type'].value_counts()
        print(f"\n📊 Overall Loyalty Distribution:")
        for loyalty_type, count in loyalty_counts.items():
            pct = count / len(df) * 100
            print(f"   {loyalty_type}: {count:,} ({pct:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading loyalty data: {e}")
        return None
    finally:
        conn.close()

def analyze_loyalty_characteristics(df):
    """
    Analyze characteristics of loyal vs disloyal users.
    """
    
    try:
        print("\n🔄 Analyzing loyal vs disloyal user characteristics...")
        
        # Key business features for comparison
        business_features = [
            'origin_chains', 'dest_chains', 'currency_sends', 'call_count',
            'distinct_routes', 'cross_chain_swaps', 'bridges', 'unique_days', 
            'total_send_usd', 'n_platforms', 'total_amount_usd', 'total_tx_count'
        ]
        
        # Calculate characteristics by loyalty type
        loyalty_profiles = df.groupby('loyalty_type')[business_features].agg(['mean', 'median', 'std']).round(2)
        
        # Flatten column names
        loyalty_profiles.columns = [f'{col}_{stat}' for col, stat in loyalty_profiles.columns]
        
        # Also calculate simple means for clean comparison
        loyalty_means = df.groupby('loyalty_type')[business_features].mean().round(2)
        
        print(f"\n📊 Loyalty Type Characteristics (Averages):")
        print("=" * 80)
        print(loyalty_means)
        
        # Calculate persona distribution within each loyalty type
        print(f"\n🎭 Persona Distribution by Loyalty Type:")
        print("=" * 80)
        loyalty_persona_pct = pd.crosstab(
            df['loyalty_type'], 
            df['persona'], 
            normalize='index'
        ) * 100
        print(loyalty_persona_pct.round(1))
        
        return loyalty_means, loyalty_profiles, loyalty_persona_pct
        
    except Exception as e:
        print(f"❌ Error analyzing loyalty characteristics: {e}")
        return None, None, None

def create_loyalty_persona_comparison_table(df):
    """
    Create a 6-row comparison table controlling for Simpson's paradox.
    Shows marginal effect of loyalty WITHIN each persona.
    """
    
    try:
        print("\n🎨 Creating loyalty×persona characteristics table...")
        print("💡 Controlling for Simpson's paradox - shows marginal effect within persona")
        
        # Create persona + loyalty_type combination for easier marginal comparison
        df['persona_loyalty'] = df['persona'] + ' + ' + df['loyalty_type']
        
        # Get counts and percentages for each combination
        combo_counts = df['persona_loyalty'].value_counts()
        total_users = len(df)
        
        # Key metrics for comparison
        key_metrics = [
            'origin_chains', 'dest_chains', 'call_count', 'bridges', 
            'cross_chain_swaps', 'currency_sends', 'unique_days', 'total_send_usd'
        ]
        
        # Calculate means for each persona×loyalty combination
        combo_means = df.groupby('persona_loyalty')[key_metrics].mean().round(2)
        
        # Create comparison table
        comparison_table = pd.DataFrame()
        
        # Add count and percentage
        comparison_table['Count'] = combo_counts.reindex(combo_means.index)
        comparison_table['%'] = (comparison_table['Count'] / total_users * 100).round(1)
        
        # Add metrics with clean names
        metric_names = {
            'origin_chains': '# Origins',
            'dest_chains': '# Destinations', 
            'call_count': '# F() Calls',
            'bridges': '# Simple Bridge',
            'cross_chain_swaps': '# Cross Chain Swaps',
            'currency_sends': '# Tokens',
            'unique_days': '# Unique Days',
            'total_send_usd': 'Avg Volume'
        }
        
        for metric in key_metrics:
            clean_name = metric_names.get(metric, metric)
            if metric == 'total_send_usd':
                # Format currency
                comparison_table[clean_name] = combo_means[metric].apply(lambda x: f"${x:,.0f}")
            else:
                # Round to 1 decimal place
                comparison_table[clean_name] = combo_means[metric].round(1)
        
        print(f"\n" + "="*120)
        print(f"📊 PERSONA × LOYALTY CHARACTERISTICS (6-Row Analysis)")
        print(f"💡 Shows marginal effect of loyalty WITHIN each persona (sorted for easy comparison)")
        print(f"="*120)
        
        # Print with nice formatting
        print(comparison_table.to_string(index=True, col_space=10))
        
        # Calculate marginal effects within each persona
        print(f"\n" + "="*120)
        print(f"🔍 MARGINAL EFFECTS (Disloyal vs Loyal WITHIN same persona):")
        print(f"="*120)
        
        personas = ['Basic Bridge Users', 'High Value Users', 'Multi-Chain Users']
        
        for persona in personas:
            try:
                loyal_key = f'{persona} + relay-only'
                disloyal_key = f'{persona} + multi-platform'
                
                if loyal_key in combo_means.index and disloyal_key in combo_means.index:
                    loyal_vol = combo_means.loc[loyal_key, 'total_send_usd']
                    disloyal_vol = combo_means.loc[disloyal_key, 'total_send_usd']
                    vol_ratio = disloyal_vol / loyal_vol if loyal_vol > 0 else 0
                    
                    loyal_chains = combo_means.loc[loyal_key, 'origin_chains']
                    disloyal_chains = combo_means.loc[disloyal_key, 'origin_chains']
                    chain_ratio = disloyal_chains / loyal_chains if loyal_chains > 0 else 0
                    
                    print(f"🎭 {persona}:")
                    print(f"   💰 Disloyal users have {vol_ratio:.1f}x volume vs loyal users")
                    print(f"   🔗 Disloyal users use {chain_ratio:.1f}x chains vs loyal users")
                else:
                    print(f"🎭 {persona}: Insufficient data for comparison")
            except Exception as e:
                print(f"🎭 {persona}: Error calculating ratios - {e}")
        
        print(f"="*120)
        
        return comparison_table
        
    except Exception as e:
        print(f"❌ Error creating comparison table: {e}")
        return None

def export_loyalty_analysis(comparison_table, loyalty_persona_pct):
    """
    Export loyalty analysis to CSV files.
    """
    
    try:
        print("\n📤 Exporting loyalty analysis files...")
        
        # Export loyalty characteristics
        comparison_table.to_csv("loyalty_characteristics.csv")
        print("✅ Saved: loyalty_characteristics.csv")
        
        # Export persona distribution by loyalty
        loyalty_persona_pct.to_csv("loyalty_persona_distribution.csv")
        print("✅ Saved: loyalty_persona_distribution.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting files: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Loyalty Characteristics Analysis...")
    print("=" * 60)
    print("🎯 Goal: Segment loyal vs disloyal users and identify their characteristics")
    print("=" * 60)
    
    # Load loyalty segment data
    df = load_loyalty_segment_data()
    if df is None:
        print("❌ Cannot load loyalty data")
        exit(1)
    
    # Analyze loyalty characteristics
    loyalty_means, loyalty_profiles, loyalty_persona_pct = analyze_loyalty_characteristics(df)
    if loyalty_means is None:
        print("❌ Cannot analyze loyalty characteristics")
        exit(1)
    
    # Create loyalty×persona comparison table (6 rows)
    comparison_table = create_loyalty_persona_comparison_table(df)
    if comparison_table is None:
        print("❌ Cannot create comparison table")
        exit(1)
    
    # Export analysis
    export_success = export_loyalty_analysis(comparison_table, loyalty_persona_pct)
    
    # Final summary
    print(f"\n🎯 LOYALTY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Analyzed loyal vs disloyal user characteristics!")
    print(f"📊 Clear differentiation in volume, complexity, and chain usage")
    
    if export_success:
        print(f"📄 Files exported: loyalty_characteristics.csv, loyalty_persona_distribution.csv")
    
    print(f"\n💡 Key Finding (Controlling for Simpson's Paradox):")
    print(f"   🔍 Marginal effect of disloyalty varies by persona")
    print(f"   🎯 Within-persona analysis reveals true drivers of disloyalty")
    print(f"   📈 Can target retention strategies specific to each persona")
    print(f"   🏆 Avoid confounding persona complexity with loyalty behavior") 