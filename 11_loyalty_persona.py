import sqlite3
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_wallet_features_from_db(db_name="relay_analysis.db"):
    """
    Load wallet features directly from relay_transactions, filtered to loyalty users.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Calculating wallet features from relay_transactions...")
        
        # Calculate features on-the-fly, joining with loyalty_aggregate to ensure same user set
        features_query = """
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
        wallet_features AS (
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
            WHERE r.wallet NOT IN (SELECT wallet FROM exclusions)
                AND r.created_at >= '2025-01-01'
            GROUP BY r.wallet
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
            total_send_usd,
            -- Log transformations for heavy-tailed features
            LOG10(CASE WHEN call_count > 0 THEN call_count ELSE 1 END) as call_count_log,
            LOG10(CASE WHEN cross_chain_swaps > 0 THEN cross_chain_swaps ELSE 1 END) as cross_chain_swaps_log,
            LOG10(CASE WHEN bridges > 0 THEN bridges ELSE 1 END) as bridges_log,
            LOG10(CASE WHEN total_send_usd > 0 THEN total_send_usd ELSE 1 END) as total_send_usd_log
        FROM wallet_features
        """
        
        features_df = pd.read_sql(features_query, conn)
        
        print(f"📊 Calculated features for {len(features_df):,} wallets")
        print(f"   Features: origin_chains, dest_chains, currency_sends, call_count, distinct_routes,")
        print(f"            cross_chain_swaps, bridges, unique_days, total_send_usd + log versions")
        
        return features_df
        
    except Exception as e:
        print(f"❌ Error calculating wallet features: {e}")
        return None
    finally:
        conn.close()

def perform_pca_transformation(features_df):
    """
    Perform PCA transformation on wallet features for clustering.
    """
    
    try:
        print("🔄 Performing PCA transformation...")
        
        # Select features for PCA (use log versions for heavy-tailed features)
        clustering_features = [
            'origin_chains', 'dest_chains', 'currency_sends', 'call_count_log', 
            'distinct_routes', 'cross_chain_swaps_log', 'bridges_log', 
            'unique_days', 'total_send_usd_log'
        ]
        
        # Extract feature matrix
        X = features_df[clustering_features].fillna(0)
        
        # Standardize features before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=5)  # Keep top 5 components
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA results dataframe
        pca_df = pd.DataFrame({
            'wallet': features_df['wallet'].values,
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1], 
            'PC3': X_pca[:, 2],
            'PC4': X_pca[:, 3],
            'PC5': X_pca[:, 4]
        })
        
        # Show PCA explained variance
        print(f"\n📊 PCA Explained Variance:")
        cumulative_variance = 0
        for i, var in enumerate(pca.explained_variance_ratio_):
            cumulative_variance += var
            print(f"   PC{i+1}: {var:.3f} ({var*100:.1f}%) - Cumulative: {cumulative_variance:.3f} ({cumulative_variance*100:.1f}%)")
        
        print(f"✅ PCA transformation complete: {pca_df.shape}")
        
        return pca_df, pca, scaler, clustering_features
        
    except Exception as e:
        print(f"❌ Error performing PCA: {e}")
        return None, None, None, None

def perform_kmeans_clustering(pca_df, n_clusters=3):
    """
    Perform k-means clustering on PCA components to find natural user groups.
    """
    
    try:
        print(f"🎯 Performing k-means clustering with {n_clusters} clusters...")
        
        # Use top 3 PCA components for clustering
        X_pca = pca_df[['PC1', 'PC2', 'PC3']].values
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        
        # Add cluster labels to dataframe
        pca_df['cluster_id'] = cluster_labels
        
        # Calculate cluster centroids and sizes
        cluster_centroids = pca_df.groupby('cluster_id')[['PC1', 'PC2', 'PC3']].mean()
        cluster_sizes = pca_df['cluster_id'].value_counts().sort_index()
        
        print(f"\n📊 Cluster Results:")
        print(f"   Cluster Centroids (PCA space):")
        print(cluster_centroids.round(3))
        
        print(f"\n   Cluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            pct = size / len(pca_df) * 100
            print(f"   Cluster {cluster_id}: {size:,} wallets ({pct:.1f}%)")
        
        return pca_df, kmeans, cluster_centroids
        
    except Exception as e:
        print(f"❌ Error performing k-means clustering: {e}")
        return None, None, None

def map_clusters_to_personas(features_df, pca_df):
    """
    Map k-means clusters to business-interpretable personas based on actual behavior profiles.
    """
    
    try:
        print("🏷️ Mapping clusters to business personas...")
        
        # Merge original features with cluster assignments AND PCA scores
        pca_columns = [col for col in pca_df.columns if col.startswith('PC')]
        merge_columns = ['wallet', 'cluster_id'] + pca_columns
        
        cluster_features_df = features_df.merge(
            pca_df[merge_columns], 
            on='wallet', 
            how='inner'
        )
        
        print(f"✅ Merged features with PCA scores: {pca_columns}")
        
        # Calculate average behavior profiles for each cluster
        business_features = [
            'origin_chains', 'dest_chains', 'currency_sends', 'call_count',
            'distinct_routes', 'cross_chain_swaps', 'bridges', 'unique_days', 'total_send_usd'
        ]
        
        cluster_profiles = cluster_features_df.groupby('cluster_id')[business_features].mean().round(2)
        
        print(f"\n📈 Cluster Business Profiles:")
        print(cluster_profiles)
        
                 # Get cluster centroids in PCA space for interpretation
        pca_centroids = pca_df.groupby('cluster_id')[['PC1', 'PC2', 'PC3']].mean()
        print(f"\n📊 PCA Centroids for Interpretation:")
        print(pca_centroids.round(3))
        
        # Interpret clusters based on PCA centroid positions
        # Use your PCA component interpretations:
        # PC1 (64.7%): Multi-chain/complex usage 
        # PC2 (11.0%): High value - specific usage (limited destinations)
        # PC3 (8.7%): Simple bridging, not engaged in complex swaps
        
        # Manual interpretation based on actual centroid positions
        # Look at the centroids and assign personas based on PCA interpretations
        cluster_personas = {}
        
        # Sort clusters by PC1 (primary variance component) 
        pc1_sorted = pca_centroids.sort_values('PC1', ascending=False)
        
        print(f"\n🔍 Cluster Interpretation Logic:")
        for i, (cluster_id, row) in enumerate(pc1_sorted.iterrows()):
            print(f"   Cluster {cluster_id}: PC1={row['PC1']:.3f}, PC2={row['PC2']:.3f}, PC3={row['PC3']:.3f}")
            
            if i == 0:  # Highest PC1
                cluster_personas[cluster_id] = "Multi-Chain Users"
                print(f"     → Multi-Chain Users (highest PC1 = most multi-chain/complex)")
            elif row['PC1'] > 0:  # Positive PC1 but not highest
                cluster_personas[cluster_id] = "High Value Users" 
                print(f"     → High Value Users (moderate PC1 = some complexity)")
            else:  # Low/negative PC1
                cluster_personas[cluster_id] = "Basic Bridge Users"
                print(f"     → Basic Bridge Users (low PC1 = limited complexity)")
        
        def interpret_cluster(cluster_id):
            return cluster_personas[cluster_id]

        # Create cluster-to-persona mapping
        cluster_mapping = {}
        for cluster_id in cluster_profiles.index:
            persona = interpret_cluster(cluster_id)
            cluster_mapping[cluster_id] = persona
            
            size = len(cluster_features_df[cluster_features_df['cluster_id'] == cluster_id])
            pct = size / len(cluster_features_df) * 100
            print(f"   Cluster {cluster_id} → {persona}: {size:,} wallets ({pct:.1f}%)")
        
        # Apply persona labels
        cluster_features_df['persona'] = cluster_features_df['cluster_id'].map(cluster_mapping)
        
        # Show final persona distribution
        print(f"\n📊 Final Persona Distribution:")
        persona_counts = cluster_features_df['persona'].value_counts()
        for persona, count in persona_counts.items():
            pct = count / len(cluster_features_df) * 100
            print(f"   {persona}: {count:,} ({pct:.1f}%)")
        
        # Show persona characteristics
        print(f"\n📈 Persona Characteristics (Average Values):")
        persona_profiles = cluster_features_df.groupby('persona')[business_features].mean().round(2)
        print(persona_profiles)
        
        return cluster_features_df, cluster_mapping
        
    except Exception as e:
        print(f"❌ Error mapping clusters to personas: {e}")
        return None, None

def get_loyalty_classification(db_name="relay_analysis.db"):
    """
    Get loyalty classification from the loyalty_aggregate table.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Loading loyalty classification...")
        
        loyalty_query = """
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
        """
        
        loyalty_df = pd.read_sql(loyalty_query, conn)
        
        print(f"📊 Loyalty Classification Results:")
        loyalty_summary = loyalty_df['loyalty_type'].value_counts()
        for loyalty_type, count in loyalty_summary.items():
            pct = count / len(loyalty_df) * 100
            print(f"   {loyalty_type}: {count:,} ({pct:.1f}%)")
        
        return loyalty_df
        
    except Exception as e:
        print(f"❌ Error loading loyalty classification: {e}")
        return None
    finally:
        conn.close()

def create_loyalty_persona_analysis(persona_df, loyalty_df):
    """
    Combine persona and loyalty data for comprehensive analysis.
    """
    
    try:
        print("🔄 Combining persona and loyalty data...")
        
        # Merge on wallet (handle different column names)
        merged_df = persona_df.merge(
            loyalty_df, 
            left_on='wallet', 
            right_on='wallet_', 
            how='inner'
        )
        
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

def save_results(merged_df, db_name="relay_analysis.db"):
    """
    Save results to database and CSV files.
    """
    
    try:
        print("💾 Saving results...")
        
        # Save to SQLite
        conn = sqlite3.connect(db_name)
        
        # Select key columns for the table including PCA scores
        pca_columns = [col for col in merged_df.columns if col.startswith('PC')]
        
        base_columns = [
            'wallet_', 'persona', 'cluster_id', 'loyalty_type', 'n_platforms', 
            'n_source_chains', 'n_dest_chains', 'total_amount_usd', 'total_tx_count',
            'origin_chains', 'dest_chains', 'currency_sends', 'total_send_usd'
        ]
        
        # Include PCA scores if they exist
        all_columns = base_columns + pca_columns
        available_columns = [col for col in all_columns if col in merged_df.columns]
        
        table_df = merged_df[available_columns].copy()
        
        print(f"💾 Saving {len(available_columns)} columns including {len(pca_columns)} PCA components")
        
        table_df.to_sql('loyalty_persona_final', conn, if_exists='replace', index=False)
        
        # Create indexes
        conn.execute("CREATE INDEX idx_loyalty_persona_wallet ON loyalty_persona_final(wallet_);")
        conn.execute("CREATE INDEX idx_loyalty_persona_persona ON loyalty_persona_final(persona);")
        conn.execute("CREATE INDEX idx_loyalty_persona_loyalty ON loyalty_persona_final(loyalty_type);")
        
        conn.close()
        
        # Export CSV files
        merged_df.to_csv("loyalty_persona_analysis.csv", index=False)
        
        # Cross-tabulation summaries
        loyalty_persona_crosstab = pd.crosstab(
            merged_df['persona'], 
            merged_df['loyalty_type'], 
            margins=True
        )
        loyalty_persona_crosstab.to_csv("loyalty_persona_crosstab.csv")
        
        # Persona characteristics
        business_features = [
            'origin_chains', 'dest_chains', 'currency_sends', 'call_count',
            'distinct_routes', 'cross_chain_swaps', 'bridges', 'unique_days', 'total_send_usd'
        ]
        
        persona_summary = merged_df.groupby('persona').agg({
            'wallet_': 'count',
            'loyalty_type': lambda x: (x == 'multi-platform').mean() * 100,
            'total_amount_usd': ['mean', 'median'],
            'total_tx_count': ['mean', 'median'],
            'n_platforms': 'mean',
            **{col: 'mean' for col in business_features}
        }).round(2)
        
        persona_summary.to_csv("persona_characteristics.csv")
        
        print(f"✅ Results saved:")
        print(f"   - Database table: loyalty_persona_final")
        print(f"   - loyalty_persona_analysis.csv")
        print(f"   - loyalty_persona_crosstab.csv")
        print(f"   - persona_characteristics.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Loyalty + Persona Analysis...")
    print("=" * 60)
    
    # Step 1: Load wallet features
    features_df = load_wallet_features_from_db()
    if features_df is None:
        print("❌ Cannot proceed without wallet features")
        exit(1)
    
    # Step 2: PCA transformation for clean clustering space
    pca_df, pca, scaler, clustering_features = perform_pca_transformation(features_df)
    if pca_df is None:
        print("❌ Cannot proceed without PCA transformation")
        exit(1)
    
    # Step 3: K-means clustering for natural groupings
    pca_df, kmeans, cluster_centroids = perform_kmeans_clustering(pca_df)
    if pca_df is None:
        print("❌ Cannot proceed without k-means clustering")
        exit(1)
    
    # Step 4: Map clusters to business personas
    persona_df, cluster_mapping = map_clusters_to_personas(features_df, pca_df)
    if persona_df is None:
        print("❌ Cannot proceed without persona mapping")
        exit(1)
    
    # Step 5: Load loyalty classification
    loyalty_df = get_loyalty_classification()
    if loyalty_df is None:
        print("❌ Cannot proceed without loyalty data")
        exit(1)
    
    # Step 6: Combine loyalty + persona analysis
    merged_df = create_loyalty_persona_analysis(persona_df, loyalty_df)
    if merged_df is None:
        print("❌ Cannot proceed without merged analysis")
        exit(1)
    
    # Step 7: Save results
    save_success = save_results(merged_df)
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Loyalty + Persona analysis complete!")
    print(f"📊 Analyzed {len(merged_df):,} wallets")
    print(f"🎭 {merged_df['persona'].nunique()} distinct personas identified")
    print(f"🔄 {merged_df['loyalty_type'].nunique()} loyalty types analyzed")
    
    # Key insights
    multi_platform_pct = (merged_df['loyalty_type'] == 'multi-platform').mean() * 100
    print(f"\n💡 Key Insights:")
    print(f"   📈 {multi_platform_pct:.1f}% of users are multi-platform")
    
    top_persona = merged_df['persona'].value_counts().index[0]
    top_persona_pct = merged_df['persona'].value_counts().iloc[0] / len(merged_df) * 100
    print(f"   🎭 Most common persona: {top_persona} ({top_persona_pct:.1f}%)")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Review persona characteristics for business strategy")
    print(f"   2. Analyze loyalty patterns for retention opportunities")
    print(f"   3. Develop persona-specific product features")
    print(f"   4. Build targeted marketing campaigns by persona")
