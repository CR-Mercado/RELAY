import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def load_pca_components(csv_path="pca_component_loadings.csv"):
    """
    Load PCA component loadings to transform wallet features into persona scores.
    """
    try:
        components_df = pd.read_csv(csv_path, index_col=0)
        print(f"✅ Loaded PCA components: {components_df.shape}")
        print(f"   Features: {list(components_df.index)}")
        print(f"   Components: {list(components_df.columns)}")
        return components_df
    except Exception as e:
        print(f"❌ Error loading PCA components: {e}")
        return None

def get_loyalty_classification(db_name="relay_analysis.db"):
    """
    Get loyalty classification (relay-only vs multi-platform) with user metrics.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Calculating loyalty classification...")
        
        loyalty_query = """
        WITH loyalty_flag AS (
            SELECT 
                wallet_, 
                CASE 
                    WHEN COUNT(DISTINCT platform_) = 1 THEN 'relay-only'
                    ELSE 'multi-platform'
                END as loyalty_type,
                COUNT(DISTINCT platform_) as n_platforms,
                COUNT(DISTINCT source_chain) as n_source_chains,
                COUNT(DISTINCT destination_chain) as n_dest_chains,
                SUM(amount_usd) as total_amount_usd,
                SUM(tx_count) as total_tx_count
            FROM loyalty_aggregate 
            GROUP BY wallet_
        )
        SELECT * FROM loyalty_flag
        """
        
        loyalty_df = pd.read_sql(loyalty_query, conn)
        
        print(f"📊 Loyalty Classification Results:")
        loyalty_summary = loyalty_df['loyalty_type'].value_counts()
        for loyalty_type, count in loyalty_summary.items():
            pct = count / len(loyalty_df) * 100
            print(f"   {loyalty_type}: {count:,} ({pct:.1f}%)")
        
        return loyalty_df
        
    except Exception as e:
        print(f"❌ Error calculating loyalty classification: {e}")
        return None
    finally:
        conn.close()

def get_wallet_features(db_name="relay_analysis.db"):
    """
    Calculate wallet features on-the-fly from relay_transactions, filtered to loyalty users.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Calculating wallet features from relay_transactions...")
        
        # Calculate features on-the-fly, joining with loyalty_aggregate to ensure same user set
        features_query = """
                 -- loyalty_aggregate already filters exclusions out 
         WITH wallet_raw_features AS (
             SELECT 
                 r.wallet,
                 COUNT(DISTINCT r.origin_chain_name) as origin_chains,
                 COUNT(DISTINCT r.destination_chain_name) as dest_chains,
                 COUNT(DISTINCT r.user_send_currency) as currency_sends,
                 SUM(CASE WHEN r.is_call = 1 THEN 1 ELSE 0 END) as call_count,
                 COUNT(DISTINCT r.route_source) as distinct_routes,
                 SUM(CASE WHEN r.execution_kind = 'Cross chain swap' THEN 1 ELSE 0 END) as cross_chain_swaps,
                 SUM(CASE WHEN r.execution_kind = 'Bridge' THEN 1 ELSE 0 END) as bridges,
                 COUNT(DISTINCT DATE(r.created_at)) as unique_days,
                 SUM(r.user_send_currency_usd) as total_send_usd
             FROM relay_transactions r
             INNER JOIN (SELECT DISTINCT wallet_ FROM loyalty_aggregate) l ON r.wallet = l.wallet_
             GROUP BY r.wallet
        )
        SELECT 
            wallet,
            origin_chains,
            dest_chains,
            currency_sends,
            LOG10(CASE WHEN call_count > 0 THEN call_count ELSE 1 END) as call_count_log,
            distinct_routes,
            LOG10(CASE WHEN cross_chain_swaps > 0 THEN cross_chain_swaps ELSE 1 END) as cross_chain_swaps_log,
            LOG10(CASE WHEN bridges > 0 THEN bridges ELSE 1 END) as bridges_log,
            unique_days,
            LOG10(CASE WHEN total_send_usd > 0 THEN total_send_usd ELSE 1 END) as total_send_usd_log
        FROM wallet_raw_features
        """
        
        features_df = pd.read_sql(features_query, conn)
        
        print(f"📊 Calculated features for {len(features_df):,} wallets")
        print(f"   Features: origin_chains, dest_chains, currency_sends, call_count_log,")
        print(f"            distinct_routes, cross_chain_swaps_log, bridges_log,") 
        print(f"            unique_days, total_send_usd_log")
        
        return features_df
        
    except Exception as e:
        print(f"❌ Error calculating wallet features: {e}")
        return None
    finally:
        conn.close()

def transform_to_pca_scores(features_df, components_df):
    """
    Transform wallet features to PCA scores using component loadings.
    """
    
    try:
        print("🔄 Transforming features to PCA scores...")
        
        # Ensure feature order matches component loadings
        feature_cols = list(components_df.index)
        missing_features = [f for f in feature_cols if f not in features_df.columns]
        
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return None
        
        # Extract feature matrix in correct order
        X = features_df[feature_cols].values
        
        # Transform to PCA scores: X @ components
        pca_scores = X @ components_df.values
        
        # Create PCA scores DataFrame
        pca_df = pd.DataFrame(
            pca_scores,
            columns=[f'PC{i+1}' for i in range(len(components_df.columns))],
            index=features_df.index
        )
        
        # Add wallet identifier
        pca_df['wallet'] = features_df['wallet'].values
        
        print(f"✅ PCA transformation complete: {pca_df.shape}")
        
        return pca_df
        
    except Exception as e:
        print(f"❌ Error transforming to PCA scores: {e}")
        return None

def assign_personas(pca_df, n_clusters=3):
    """
    Assign personas based on highest relative ranking across top 3 PCA dimensions.
    Each wallet gets assigned to the cluster where it has the strongest relative position.
    """
    
    try:
        print(f"🎭 Assigning {n_clusters} personas using relative ranking approach...")
        
        # Calculate percentile ranks for top 3 PCA components by variance
        pca_df['PC1_rank'] = pca_df['PC1'].rank(pct=True)  # 64.7% variance
        pca_df['PC2_rank'] = pca_df['PC2'].rank(pct=True)  # 11.0% variance  
        pca_df['PC3_rank'] = pca_df['PC3'].rank(pct=True)  # 8.7% variance
        
        # Assign each wallet to the cluster where it has the highest relative rank
        def classify_persona(row):
            ranks = {
                'PC1_rank': row['PC1_rank'],
                'PC2_rank': row['PC2_rank'], 
                'PC3_rank': row['PC3_rank']
            }
            
            # Find the PCA dimension where this wallet ranks highest
            highest_rank_pc = max(ranks, key=ranks.get)
            
            # Map to persona based on strongest PCA dimension
            if highest_rank_pc == 'PC1_rank':
                return "Multi-Chain Users"      # Strongest in multi-chain engagement
            elif highest_rank_pc == 'PC2_rank':
                return "High Value Users"       # Strongest in high value/low frequency
            else:  # PC3_rank
                return "Basic Bridge Users"           # Strongest in bridge-focused behavior
        
        # Apply persona classification
        pca_df['persona'] = pca_df.apply(classify_persona, axis=1)
        
        # Show persona distribution
        print(f"\n📊 Persona Distribution (by strongest PCA dimension):")
        persona_counts = pca_df['persona'].value_counts()
        for persona, count in persona_counts.items():
            pct = count / len(pca_df) * 100
            print(f"   {persona}: {count:,} ({pct:.1f}%)")
        
        # Show cluster characteristics
        print(f"\n📈 Cluster Characteristics:")
        print("   Average PCA Scores by Persona:")
        cluster_stats = pca_df.groupby('persona')[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].mean().round(3)
        print(cluster_stats)
        
        print(f"\n   Average Percentile Ranks by Persona:")
        rank_stats = pca_df.groupby('persona')[['PC1_rank', 'PC2_rank', 'PC3_rank']].mean().round(3)
        print(rank_stats)
        
        # Show validation: each persona should be strongest in its target dimension
        print(f"\n✅ Validation - Average ranks by persona (should be highest in target PC):")
        for persona in persona_counts.index:
            persona_data = pca_df[pca_df['persona'] == persona]
            avg_ranks = persona_data[['PC1_rank', 'PC2_rank', 'PC3_rank']].mean()
            strongest_pc = avg_ranks.idxmax()
            print(f"   {persona}: Strongest in {strongest_pc} (avg rank: {avg_ranks[strongest_pc]:.3f})")
        
        return pca_df
        
    except Exception as e:
        print(f"❌ Error assigning personas: {e}")
        return None

def create_loyalty_persona_analysis(loyalty_df, persona_df):
    """
    Combine loyalty classification with persona assignment for comprehensive analysis.
    """
    
    try:
        print("🔄 Combining loyalty and persona data...")
        
        # Merge on wallet (handle different column names)
        if 'wallet_' in loyalty_df.columns and 'wallet' in persona_df.columns:
            merged_df = loyalty_df.merge(
                persona_df, 
                left_on='wallet_', 
                right_on='wallet', 
                how='inner'
            )
        else:
            merged_df = loyalty_df.merge(persona_df, on='wallet', how='inner')
        
        print(f"✅ Merged data: {len(merged_df):,} wallets")
        
        # Cross-tabulation analysis
        print(f"\n📊 LOYALTY × PERSONA CROSS-ANALYSIS:")
        print("=" * 60)
        
        # Loyalty by Persona
        loyalty_persona_crosstab = pd.crosstab(
            merged_df['persona'], 
            merged_df['loyalty_type'], 
            margins=True
        )
        print(f"\n🎭 Personas by Loyalty Type:")
        print(loyalty_persona_crosstab)
        
        # Calculate percentages
        loyalty_persona_pct = pd.crosstab(
            merged_df['persona'], 
            merged_df['loyalty_type'], 
            normalize='index'
        ) * 100
        print(f"\n📈 Loyalty Distribution by Persona (%):")
        print(loyalty_persona_pct.round(1))
        
        # Volume analysis by loyalty + persona
        print(f"\n💰 Volume Analysis by Loyalty + Persona:")
        volume_analysis = merged_df.groupby(['loyalty_type', 'persona']).agg({
            'total_amount_usd': ['count', 'mean', 'sum'],
            'total_tx_count': ['mean', 'sum'],
            'n_platforms': 'mean'
        }).round(2)
        print(volume_analysis)
        
        return merged_df
        
    except Exception as e:
        print(f"❌ Error creating loyalty persona analysis: {e}")
        return None

def save_loyalty_persona_table(merged_df, db_name="relay_analysis.db", table_name="loyalty_persona"):
    """
    Save the combined loyalty + persona data to SQLite table.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print(f"💾 Saving loyalty persona table: {table_name}")
        
        # Select key columns for the table
        table_df = merged_df[[
            'wallet_', 'loyalty_type', 'persona', 'n_platforms', 
            'n_source_chains', 'n_dest_chains', 'total_amount_usd', 'total_tx_count',
            'PC1', 'PC2', 'PC3', 'PC4', 'PC5'
        ]].copy()
        
        # Save to SQLite
        table_df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Create indexes
        conn.execute(f"CREATE INDEX idx_{table_name}_wallet ON {table_name}(wallet_);")
        conn.execute(f"CREATE INDEX idx_{table_name}_loyalty ON {table_name}(loyalty_type);")
        conn.execute(f"CREATE INDEX idx_{table_name}_persona ON {table_name}(persona);")
        
        print(f"✅ Table saved: {len(table_df):,} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving loyalty persona table: {e}")
        return False
    finally:
        conn.close()

def export_analysis_files(merged_df):
    """
    Export analysis to CSV files.
    """
    
    try:
        print("📤 Exporting analysis files...")
        
        # Main results
        merged_df.to_csv("loyalty_persona_analysis.csv", index=False)
        
        # Cross-tabulation summaries
        loyalty_persona_crosstab = pd.crosstab(
            merged_df['persona'], 
            merged_df['loyalty_type'], 
            margins=True
        )
        loyalty_persona_crosstab.to_csv("loyalty_persona_crosstab.csv")
        
        # Persona characteristics
        persona_summary = merged_df.groupby('persona').agg({
            'wallet_': 'count',
            'loyalty_type': lambda x: (x == 'multi-platform').mean() * 100,
            'total_amount_usd': ['mean', 'median'],
            'total_tx_count': ['mean', 'median'],
            'n_platforms': 'mean',
            'PC1': 'mean',
            'PC2': 'mean', 
            'PC3': 'mean',
            'PC4': 'mean',
            'PC5': 'mean'
        }).round(2)
        
        persona_summary.columns = [
            'wallet_count', 'multi_platform_pct', 'avg_volume', 'median_volume',
            'avg_tx_count', 'median_tx_count', 'avg_platforms',
            'avg_PC1', 'avg_PC2', 'avg_PC3', 'avg_PC4', 'avg_PC5'
        ]
        persona_summary.to_csv("persona_characteristics.csv")
        
        print(f"💾 Files exported:")
        print(f"   - loyalty_persona_analysis.csv")
        print(f"   - loyalty_persona_crosstab.csv")  
        print(f"   - persona_characteristics.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting files: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Loyalty + Persona Analysis...")
    print("=" * 60)
    
    # Load PCA components
    components_df = load_pca_components()
    if components_df is None:
        print("❌ Cannot proceed without PCA components")
        exit(1)
    
    # Get loyalty classification
    loyalty_df = get_loyalty_classification()
    if loyalty_df is None:
        print("❌ Cannot proceed without loyalty data")
        exit(1)
    
    # Get wallet features
    features_df = get_wallet_features()
    if features_df is None:
        print("❌ Cannot proceed without wallet features")
        exit(1)
    
    # Transform to PCA scores
    pca_df = transform_to_pca_scores(features_df, components_df)
    if pca_df is None:
        print("❌ Cannot proceed without PCA transformation")
        exit(1)
    
    # Assign personas
    persona_df = assign_personas(pca_df)
    if persona_df is None:
        print("❌ Cannot proceed without persona assignment")
        exit(1)
    
    # Combine loyalty + persona analysis
    merged_df = create_loyalty_persona_analysis(loyalty_df, persona_df)
    if merged_df is None:
        print("❌ Cannot proceed without merged analysis")
        exit(1)
    
    # Save to database
    save_success = save_loyalty_persona_table(merged_df)
    
    # Export analysis files
    export_success = export_analysis_files(merged_df)
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Loyalty + Persona analysis complete!")
    print(f"📊 Analyzed {len(merged_df):,} wallets")
    print(f"🎭 {merged_df['persona'].nunique()} distinct personas identified")
    print(f"🔄 {merged_df['loyalty_type'].nunique()} loyalty types analyzed")
    
    if save_success:
        print(f"💾 SQLite table created: loyalty_persona")
    if export_success:
        print(f"📤 CSV analysis files exported")
    
    print(f"\n💡 Key Insights:")
    multi_platform_pct = (merged_df['loyalty_type'] == 'multi-platform').mean() * 100
    print(f"   📈 {multi_platform_pct:.1f}% of users are multi-platform")
    
    top_persona = merged_df['persona'].value_counts().index[0]
    top_persona_pct = merged_df['persona'].value_counts().iloc[0] / len(merged_df) * 100
    print(f"   🎭 Most common persona: {top_persona} ({top_persona_pct:.1f}%)")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Analyze persona loyalty patterns")
    print(f"   2. Develop targeted retention strategies")
    print(f"   3. Compare persona performance across platforms")
    print(f"   4. Build persona-specific product features") 