import sqlite3
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, fcluster


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

def prepare_clustering_features(df, sample_size=1000):
    """
    Prepare features for clustering with log transformation and scaling.
    Sample for dendrogram visualization to avoid memory issues.
    """
    
    # Features for clustering (removed redundant features to avoid multicollinearity)
    # - total_receive_usd: highly correlated with total_send_usd  
    # - deposit_tx_count: sum of cross_chain_swaps + bridges (redundant)
    features = ['origin_chains', 'dest_chains', 'currency_sends', 'call_count', 'distinct_routes', 
               'cross_chain_swaps', 'bridges', 'unique_days', 'total_send_usd']
    
    # Create feature dataframe
    feature_df = df[features].copy()
    
    # Log transform heavy-tailed features (add 1 to handle zeros)
    log_features = ['call_count', 'cross_chain_swaps', 'bridges', 'total_send_usd']
    
    print("🔄 Applying log transformations...")
    for feature in log_features:
        feature_df[f'{feature}_log'] = np.log1p(feature_df[feature])  # log1p = log(1+x)
        print(f"   {feature}: Original range [{feature_df[feature].min():.2f}, {feature_df[feature].max():.2f}] → Log range [{feature_df[f'{feature}_log'].min():.2f}, {feature_df[f'{feature}_log'].max():.2f}]")
    
    # Use log versions for heavy-tailed features, original for others
    clustering_features = []
    for feature in features:
        if feature in log_features:
            clustering_features.append(f'{feature}_log')
        else:
            clustering_features.append(feature)
    
    # Select final feature set
    X = feature_df[clustering_features].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Sample for dendrogram (too many points make it unreadable)
    if len(X) > sample_size:
        print(f"📊 Sampling {sample_size:,} wallets for dendrogram visualization...")
        
        # Stratified sampling based on volume to ensure representation of all user types
        volume_col = 'total_send_usd'
        
        # Define volume-based strata
        p90 = df[volume_col].quantile(0.90)
        p99 = df[volume_col].quantile(0.99)
        
        high_volume = df[df[volume_col] >= p99].index  # Top 1%
        mid_volume = df[(df[volume_col] >= p90) & (df[volume_col] < p99)].index  # 90-99%
        low_volume = df[df[volume_col] < p90].index  # Bottom 90%
        
        # Sample proportionally from each stratum
        n_high = min(int(sample_size * 0.1), len(high_volume))  # 10% of sample
        n_mid = min(int(sample_size * 0.2), len(mid_volume))   # 20% of sample  
        n_low = sample_size - n_high - n_mid                   # Remaining 70%
        
        sample_idx = []
        if n_high > 0:
            sample_idx.extend(np.random.choice(high_volume, n_high, replace=False))
        if n_mid > 0:
            sample_idx.extend(np.random.choice(mid_volume, n_mid, replace=False))
        if n_low > 0 and len(low_volume) > 0:
            sample_idx.extend(np.random.choice(low_volume, min(n_low, len(low_volume)), replace=False))
        
        X_sample = X.iloc[sample_idx].copy()
        sample_wallets = df.iloc[sample_idx]['wallet'].values
        
        print(f"   📈 Stratified sample: {n_high} high-volume, {n_mid} mid-volume, {len(sample_idx)-n_high-n_mid} low-volume")
    else:
        X_sample = X.copy()
        sample_wallets = df['wallet'].values
    
    # Simple standardization (z-score normalization)
    print("📏 Standardizing features...")
    
    # Calculate means and standard deviations
    means = X.mean()
    stds = X.std()
    
    # Standardize (subtract mean, divide by std)
    X_scaled = (X - means) / stds
    X_sample_scaled = (X_sample - means) / stds
    
    # Fill any NaN values (from zero std) with 0
    X_scaled = X_scaled.fillna(0)
    X_sample_scaled = X_sample_scaled.fillna(0)
    
    print(f"✅ Features prepared: {X_scaled.shape[0]:,} total wallets, {X_sample_scaled.shape[0]:,} in sample")
    print(f"📋 Features used: {clustering_features}")
    
    return X_scaled.values, X_sample_scaled.values, sample_wallets, clustering_features, df

def perform_hierarchical_clustering(X_sample_scaled, method='ward'):
    """
    Perform hierarchical clustering and return linkage matrix.
    """
    
    print(f"🌳 Performing hierarchical clustering with {method} linkage...")
    
    # Calculate linkage matrix
    linkage_matrix = linkage(X_sample_scaled, method=method)
    
    print(f"✅ Clustering completed")
    
    return linkage_matrix

def create_dendrogram_plotly(X_sample_scaled, sample_wallets, clustering_features, output_file="dendrogram.html"):
    """
    Create an interactive dendrogram using plotly.
    """
    
    print("🎨 Creating dendrogram visualization...")
    
    # Create dendrogram using plotly (it will compute linkage internally)
    fig = ff.create_dendrogram(
        X_sample_scaled,
        orientation='bottom',
        labels=[f"W{i}" for i in range(len(sample_wallets))]
    )
    
    fig.update_layout(
        title={
            'text': f"Wallet Hierarchical Clustering Dendrogram<br><sub>Features: {', '.join(clustering_features)}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Wallets (Sample)",
        yaxis_title="Distance",
        width=1200,
        height=600,
        font=dict(size=12)
    )
    
    # Save HTML file
    fig.write_html(output_file)
    print(f"✅ Dendrogram saved to: {output_file}")
    
    return fig

def analyze_clusters(X_scaled, linkage_matrix, df, n_clusters=3):
    """
    Analyze cluster characteristics for different cluster counts.
    """
    
    print(f"\n📊 CLUSTER ANALYSIS")
    print("=" * 60)
    
    # Get cluster assignments for different numbers of clusters
    for n in [2, 3, 4, 5]:
        clusters = fcluster(linkage_matrix, n, criterion='maxclust')
        
        # For full dataset, assign clusters based on the sample clustering
        # This is a simplification - ideally we'd cluster the full dataset
        print(f"\n🔢 {n} Clusters:")
        
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = count / len(clusters) * 100
            print(f"   Cluster {cluster_id}: {count:,} wallets ({percentage:.1f}%)")

def create_cluster_profiles(df, n_clusters=3):
    """
    Create business-interpretable cluster profiles using percentile cutoffs.
    This gives us interpretable segments regardless of the clustering algorithm.
    """
    
    print(f"\n💼 BUSINESS CLUSTER PROFILES")
    print("=" * 60)
    
    # Define segments based on volume (P90/P99 cutoffs from summary stats)
    df_analysis = df.copy()
    
    # Volume-based segmentation (from your business intuition)
    p90_volume = df_analysis['total_send_usd'].quantile(0.90)  # ~$343
    p99_volume = df_analysis['total_send_usd'].quantile(0.99)  # ~$6009
    
    def assign_segment(row):
        if row['total_send_usd'] >= p99_volume:
            return 'Programmatic (Top 1%)'
        elif row['total_send_usd'] >= p90_volume:
            return 'Core Users (Top 10%)'
        else:
            return 'The Masses (Bottom 90%)'
    
    df_analysis['business_segment'] = df_analysis.apply(assign_segment, axis=1)
    
    # Analyze segments (using the same features as clustering)
    segments = df_analysis.groupby('business_segment').agg({
        'wallet': 'count',
        'total_send_usd': ['mean', 'median'],
        'unique_days': ['mean', 'median'], 
        'origin_chains': ['mean', 'median'],
        'dest_chains': ['mean', 'median'],
        'currency_sends': ['mean', 'median'],
        'call_count': ['mean', 'median'],
        'cross_chain_swaps': ['mean', 'median'],
        'bridges': ['mean', 'median'],
        'distinct_routes': ['mean', 'median']
    }).round(2)
    
    print("\n📈 Segment Characteristics:")
    print(segments)
    
    # Save segment analysis
    segments.to_csv("business_segments_analysis.csv")
    print(f"\n💾 Segment analysis saved to: business_segments_analysis.csv")
    
    return df_analysis

if __name__ == "__main__":
    # Load data
    df = load_wallet_features()
    
    if df is not None and len(df) > 0:
        
        # Prepare features for clustering
        X_scaled, X_sample_scaled, sample_wallets, clustering_features, df_full = prepare_clustering_features(df)
        
        # Perform hierarchical clustering
        linkage_matrix = perform_hierarchical_clustering(X_sample_scaled)
        
        # Create dendrogram
        fig = create_dendrogram_plotly(X_sample_scaled, sample_wallets, clustering_features)
        
        # Analyze clusters
        analyze_clusters(X_scaled, linkage_matrix, df_full)
        
        # Create business-interpretable segments
        df_with_segments = create_cluster_profiles(df_full)
        
        print(f"\n🎯 SUMMARY")
        print("=" * 60)
        print(f"✅ Hierarchical clustering completed")
        print(f"📊 Dendrogram visualization: dendrogram.html")
        print(f"📈 Business segments analysis: business_segments_analysis.csv")
        print(f"\n💡 Next steps:")
        print(f"   1. Review dendrogram to identify natural cluster count")
        print(f"   2. Compare algorithmic clusters with business segments")
        print(f"   3. Refine segmentation strategy based on insights")
        
    else:
        print("❌ No data to analyze") 