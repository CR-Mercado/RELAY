import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def prepare_clustering_features(df):
    """
    Prepare features for clustering with log transformation and scaling.
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
    
    # Standardize features
    print("📏 Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Features prepared: {X_scaled.shape[0]:,} wallets, {X_scaled.shape[1]} features")
    print(f"📋 Features used: {clustering_features}")
    
    return X_scaled, clustering_features, X.index

def perform_pca_analysis(X_scaled, n_components=None):
    """
    Perform PCA analysis to reduce dimensionality and understand feature importance.
    """
    
    if n_components is None:
        n_components = min(X_scaled.shape[1], 5)  # Max 5 components or number of features
    
    print(f"🔍 Performing PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance
    print(f"\n📊 PCA Explained Variance:")
    cumulative_variance = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative_variance += var
        print(f"   PC{i+1}: {var:.3f} ({var*100:.1f}%) - Cumulative: {cumulative_variance:.3f} ({cumulative_variance*100:.1f}%)")
    
    return pca, X_pca

def analyze_pca_components(pca, clustering_features):
    """
    Analyze and display what each PCA component represents in terms of original features.
    """
    
    print(f"\n🔍 PCA COMPONENT FORMULATIONS")
    print("=" * 60)
    print("Each PC is a weighted combination of original features:")
    print("Higher absolute weights = more important for that component\n")
    
    # Create loadings dataframe
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=clustering_features
    )
    
    # Print formulations for each component
    for i in range(pca.n_components_):
        pc_name = f'PC{i+1}'
        print(f"🎯 {pc_name} ({pca.explained_variance_ratio_[i]*100:.1f}% of variance):")
        
        # Get feature loadings for this component, sorted by absolute value
        component_loadings = loadings_df[pc_name].sort_values(key=abs, ascending=False)
        
        # Show top contributing features
        for feature, loading in component_loadings.items():
            direction = "+" if loading >= 0 else "-"
            print(f"   {direction} {abs(loading):.3f} × {feature}")
        
        # Interpret the component
        top_positive = component_loadings[component_loadings > 0].head(2)
        top_negative = component_loadings[component_loadings < 0].head(2)
        
        print(f"   📝 Interpretation: Higher {pc_name} = ", end="")
        if len(top_positive) > 0:
            print(f"More {', '.join(top_positive.index)}", end="")
        if len(top_negative) > 0:
            if len(top_positive) > 0:
                print(f" + Less {', '.join(top_negative.index)}", end="")
            else:
                print(f"Less {', '.join(top_negative.index)}", end="")
        print("\n")
    
    # Save detailed loadings
    loadings_df_rounded = loadings_df.round(3)
    loadings_df_rounded.to_csv("pca_component_loadings.csv")
    print(f"💾 Detailed PCA loadings saved to: pca_component_loadings.csv")
    
    return loadings_df

def perform_kmeans_clustering(X_pca, n_clusters_range=(2, 8)):
    """
    Perform k-means clustering on PCA components and find optimal cluster count.
    """
    
    print(f"\n🎯 Finding optimal number of clusters...")
    
    inertias = []
    cluster_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)
        print(f"   k={k}: inertia={kmeans.inertia_:.2f}")
    
    # Use elbow method suggestion (can be overridden)
    # For now, default to 3 clusters but show the inertias for user decision
    optimal_k = 3
    print(f"\n🔢 Using k={optimal_k} clusters (review inertias above to adjust)")
    
    # Final clustering with optimal k
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_pca)
    
    print(f"✅ K-means clustering completed")
    
    return kmeans_final, cluster_labels, inertias

def create_pca_visualization(X_pca, cluster_labels, clustering_features, output_file="pca_clusters.html"):
    """
    Create PCA visualization showing cluster centroids as spheres (much cleaner than thousands of points).
    """
    
    print("🎨 Creating PCA cluster centroid visualization...")
    
    # Create DataFrame with all points
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': cluster_labels
    })
    
    if X_pca.shape[1] >= 3:
        plot_df['PC3'] = X_pca[:, 2]
    
    # Calculate cluster centroids and statistics
    cluster_stats = []
    unique_clusters = sorted(plot_df['Cluster'].unique())
    
    for cluster_id in unique_clusters:
        cluster_data = plot_df[plot_df['Cluster'] == cluster_id]
        
        stats = {
            'Cluster': f'Cluster {cluster_id}',
            'PC1_mean': cluster_data['PC1'].mean(),
            'PC2_mean': cluster_data['PC2'].mean(),
            'PC1_std': cluster_data['PC1'].std(),
            'PC2_std': cluster_data['PC2'].std(),
            'count': len(cluster_data),
            'percentage': len(cluster_data) / len(plot_df) * 100
        }
        
        if X_pca.shape[1] >= 3:
            stats['PC3_mean'] = cluster_data['PC3'].mean()
            stats['PC3_std'] = cluster_data['PC3'].std()
        
        cluster_stats.append(stats)
    
    centroid_df = pd.DataFrame(cluster_stats)
    
    # Create visualization
    if X_pca.shape[1] >= 3 and 'PC3_mean' in centroid_df.columns:
        # 3D scatter plot with centroids as large spheres
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1[:len(unique_clusters)]
        
        for i, (_, row) in enumerate(centroid_df.iterrows()):
            # Add cluster centroid as sphere
            fig.add_trace(go.Scatter3d(
                x=[row['PC1_mean']],
                y=[row['PC2_mean']],
                z=[row['PC3_mean']],
                mode='markers',
                marker=dict(
                    size=15,  # Fixed reasonable size
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                name=f"{row['Cluster']} ({row['count']:,} wallets, {row['percentage']:.1f}%)",
                hovertemplate=f"<b>{row['Cluster']}</b><br>" +
                             f"PC1: {row['PC1_mean']:.2f} ± {row['PC1_std']:.2f}<br>" +
                             f"PC2: {row['PC2_mean']:.2f} ± {row['PC2_std']:.2f}<br>" +
                             f"PC3: {row['PC3_mean']:.2f} ± {row['PC3_std']:.2f}<br>" +
                             f"Count: {row['count']:,} wallets ({row['percentage']:.1f}%)<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"PCA Cluster Centroids (3D)<br><sub>Features: {', '.join(clustering_features)}</sub>",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2", 
                zaxis_title="PC3"
            )
        )
        
    else:
        # 2D scatter plot with centroids as large circles
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1[:len(unique_clusters)]
        
        for i, (_, row) in enumerate(centroid_df.iterrows()):
            # Add cluster centroid as circle
            fig.add_trace(go.Scatter(
                x=[row['PC1_mean']],
                y=[row['PC2_mean']], 
                mode='markers',
                marker=dict(
                    size=12,  # Fixed reasonable size
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                name=f"{row['Cluster']} ({row['count']:,} wallets, {row['percentage']:.1f}%)",
                hovertemplate=f"<b>{row['Cluster']}</b><br>" +
                             f"PC1: {row['PC1_mean']:.2f} ± {row['PC1_std']:.2f}<br>" +
                             f"PC2: {row['PC2_mean']:.2f} ± {row['PC2_std']:.2f}<br>" +
                             f"Count: {row['count']:,} wallets ({row['percentage']:.1f}%)<extra></extra>"
            ))
            
            # Add confidence ellipse (1 standard deviation)
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_x = row['PC1_mean'] + row['PC1_std'] * np.cos(theta)
            ellipse_y = row['PC2_mean'] + row['PC2_std'] * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=ellipse_x,
                y=ellipse_y,
                mode='lines',
                line=dict(color=colors[i], width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"PCA Cluster Centroids (2D)<br><sub>Features: {', '.join(clustering_features)}</sub>",
            xaxis_title="PC1",
            yaxis_title="PC2"
        )
    
    fig.update_layout(
        width=1000,
        height=700,
        font=dict(size=12),
        hovermode='closest'
    )
    
    # Save HTML file
    fig.write_html(output_file)
    print(f"✅ PCA centroid visualization saved to: {output_file}")
    print(f"📊 Showing {len(unique_clusters)} cluster centroids instead of {len(plot_df):,} individual points")
    
    return fig

def analyze_clusters(df, cluster_labels, clustering_features):
    """
    Analyze cluster characteristics for all wallets.
    """
    
    print(f"\n📊 CLUSTER ANALYSIS")
    print("=" * 60)
    
    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Cluster sizes
    cluster_counts = df_analysis['cluster'].value_counts().sort_index()
    print(f"\n🔢 Cluster Sizes:")
    for cluster_id, count in cluster_counts.items():
        percentage = count / len(df_analysis) * 100
        print(f"   Cluster {cluster_id}: {count:,} wallets ({percentage:.1f}%)")
    
    # Analyze cluster characteristics
    numeric_features = ['origin_chains', 'dest_chains', 'currency_sends', 'call_count', 'distinct_routes', 
                       'cross_chain_swaps', 'bridges', 'unique_days', 'total_send_usd']
    
    cluster_profiles = df_analysis.groupby('cluster')[numeric_features].agg(['mean', 'median']).round(2)
    
    print(f"\n📈 Cluster Profiles:")
    print(cluster_profiles)
    
    # Save detailed analysis
    cluster_profiles.to_csv("cluster_profiles_detailed.csv")
    print(f"\n💾 Detailed cluster analysis saved to: cluster_profiles_detailed.csv")
    
    return df_analysis, cluster_profiles

def create_business_segments(df_analysis):
    """
    Create business-interpretable segments and compare with algorithmic clusters.
    """
    
    print(f"\n💼 BUSINESS SEGMENTS vs ALGORITHMIC CLUSTERS")
    print("=" * 60)
    
    # Volume-based segmentation (from your business intuition)
    p90_volume = df_analysis['total_send_usd'].quantile(0.90)
    p99_volume = df_analysis['total_send_usd'].quantile(0.99)
    
    def assign_segment(row):
        if row['total_send_usd'] >= p99_volume:
            return 'Programmatic (Top 1%)'
        elif row['total_send_usd'] >= p90_volume:
            return 'Core Users (Top 10%)'
        else:
            return 'The Masses (Bottom 90%)'
    
    df_analysis['business_segment'] = df_analysis.apply(assign_segment, axis=1)
    
    # Cross-tabulation of business segments vs algorithmic clusters
    crosstab = pd.crosstab(df_analysis['business_segment'], df_analysis['cluster'], margins=True)
    print(f"\n📊 Business Segments vs Algorithmic Clusters:")
    print(crosstab)
    
    # Save comparison
    crosstab.to_csv("business_vs_algorithmic_clusters.csv")
    print(f"\n💾 Comparison saved to: business_vs_algorithmic_clusters.csv")
    
    return df_analysis

if __name__ == "__main__":
    # Load data
    df = load_wallet_features()
    
    if df is not None and len(df) > 0:
        
        # Prepare features for clustering
        X_scaled, clustering_features, wallet_indices = prepare_clustering_features(df)
        
        # Perform PCA analysis
        pca, X_pca = perform_pca_analysis(X_scaled)
        
        # Analyze PCA component formulations
        loadings_df = analyze_pca_components(pca, clustering_features)
        
        # Perform k-means clustering on PCA components
        kmeans, cluster_labels, inertias = perform_kmeans_clustering(X_pca)
        
        # Create PCA visualization
        fig = create_pca_visualization(X_pca, cluster_labels, clustering_features)
        
        # Analyze clusters
        df_with_clusters, cluster_profiles = analyze_clusters(df, cluster_labels, clustering_features)
        
        # Compare with business segments
        df_final = create_business_segments(df_with_clusters)
        
        print(f"\n🎯 SUMMARY")
        print("=" * 60)
        print(f"✅ PCA + K-means clustering completed")
        print(f"📊 PCA visualization: pca_clusters.html")
        print(f"🔍 PCA component loadings: pca_component_loadings.csv")
        print(f"📈 Cluster profiles: cluster_profiles_detailed.csv")
        print(f"🔍 Business comparison: business_vs_algorithmic_clusters.csv")
        print(f"\n💡 Next steps:")
        print(f"   1. Review PCA visualization to understand cluster separation")
        print(f"   2. Examine cluster profiles for business interpretation")
        print(f"   3. Compare algorithmic vs business segmentation")
        print(f"   4. Adjust cluster count if needed based on business logic")
        
    else:
        print("❌ No data to analyze") 