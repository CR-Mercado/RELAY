import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

def load_persona_data(db_name="relay_analysis.db"):
    """
    Load the persona analysis results from the database.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔍 Loading persona analysis results...")
        
        # Load the loyalty_persona_final table
        query = """
        SELECT 
            wallet_,
            persona,
            cluster_id,
            loyalty_type,
            origin_chains,
            dest_chains,
            total_send_usd
        FROM loyalty_persona_final
        """
        
        df = pd.read_sql(query, conn)
        
        print(f"✅ Loaded {len(df):,} wallets with persona assignments")
        
        # Show persona distribution
        persona_counts = df['persona'].value_counts()
        print(f"\n📊 Persona Distribution:")
        for persona, count in persona_counts.items():
            pct = count / len(df) * 100
            print(f"   {persona}: {count:,} ({pct:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading persona data: {e}")
        return None
    finally:
        conn.close()

def load_pca_scores(csv_path="loyalty_persona_analysis.csv"):
    """
    Load PCA scores from the analysis CSV file.
    """
    
    try:
        print("🔍 Loading PCA scores from CSV...")
        
        # Load the full analysis file which should have PCA scores
        df = pd.read_csv(csv_path)
        
        # Check if PCA columns exist
        pca_columns = [col for col in df.columns if col.startswith('PC')]
        
        if len(pca_columns) >= 3:
            print(f"✅ Found PCA columns: {pca_columns[:5]}")
            return df[['wallet_', 'persona', 'cluster_id'] + pca_columns[:5]]
        else:
            print(f"❌ No PCA columns found in CSV file")
            return None
            
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        # Fallback: load from database and calculate PCA on the fly
        return None

def calculate_cluster_centroids_and_stats(db_name="relay_analysis.db"):
    """
    Calculate cluster centroids and statistics from the database.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print("🔄 Loading cluster centroids from database...")
        
        # First, check what columns are available
        columns_query = "PRAGMA table_info(loyalty_persona_final)"
        columns_df = pd.read_sql(columns_query, conn)
        available_columns = columns_df['name'].tolist()
        
        # Check if PCA columns exist
        pca_columns = [col for col in available_columns if col.startswith('PC')]
        
        if len(pca_columns) < 3:
            print(f"⚠️  Only found {len(pca_columns)} PCA columns: {pca_columns}")
            print("   Need to run 11_loyalty_persona.py first to generate PCA scores")
            return calculate_cluster_centroids_fallback()
        
        print(f"✅ Found PCA columns: {pca_columns[:5]}")
        
        # Load data with PCA scores from database
        query = f"""
        SELECT 
            cluster_id,
            persona,
            {', '.join(pca_columns[:3])},
            COUNT(*) as count
        FROM loyalty_persona_final
        WHERE {pca_columns[0]} IS NOT NULL
        GROUP BY cluster_id, persona
        ORDER BY cluster_id
        """
        
        df = pd.read_sql(query, conn)
        
        if len(df) == 0:
            print("⚠️  No PCA data found in database, falling back to hardcoded values...")
            return calculate_cluster_centroids_fallback()
        
        # Calculate centroids and statistics
        cluster_stats = []
        total_wallets = df['count'].sum()
        
        for _, row in df.iterrows():
            # Get detailed stats for this cluster
            detail_query = f"""
            SELECT 
                AVG({pca_columns[0]}) as PC1_mean, AVG({pca_columns[1]}) as PC2_mean, AVG({pca_columns[2]}) as PC3_mean,
                STDEV({pca_columns[0]}) as PC1_std, STDEV({pca_columns[1]}) as PC2_std, STDEV({pca_columns[2]}) as PC3_std
            FROM loyalty_persona_final 
            WHERE cluster_id = {row['cluster_id']}
            """
            
            try:
                detail_df = pd.read_sql(detail_query, conn)
                detail = detail_df.iloc[0]
            except:
                # Fallback if STDEV function doesn't work in SQLite
                simple_query = f"""
                SELECT 
                    AVG({pca_columns[0]}) as PC1_mean, 
                    AVG({pca_columns[1]}) as PC2_mean, 
                    AVG({pca_columns[2]}) as PC3_mean
                FROM loyalty_persona_final 
                WHERE cluster_id = {row['cluster_id']}
                """
                detail_df = pd.read_sql(simple_query, conn)
                detail = detail_df.iloc[0]
                # Use default std values
                detail['PC1_std'] = 0.5
                detail['PC2_std'] = 0.3
                detail['PC3_std'] = 0.4
            
            stats = {
                'cluster_id': row['cluster_id'],
                'persona': row['persona'],
                'PC1_mean': detail['PC1_mean'] if pd.notna(detail['PC1_mean']) else 0,
                'PC2_mean': detail['PC2_mean'] if pd.notna(detail['PC2_mean']) else 0,
                'PC3_mean': detail['PC3_mean'] if pd.notna(detail['PC3_mean']) else 0,
                'PC1_std': detail.get('PC1_std', 0.5) if pd.notna(detail.get('PC1_std', 0.5)) else 0.5,
                'PC2_std': detail.get('PC2_std', 0.3) if pd.notna(detail.get('PC2_std', 0.3)) else 0.3,
                'PC3_std': detail.get('PC3_std', 0.4) if pd.notna(detail.get('PC3_std', 0.4)) else 0.4,
                'count': row['count'],
                'percentage': row['count'] / total_wallets * 100
            }
            
            # Calculate sphere size using ceiling(0.05*sqrt(# wallets))
            stats['sphere_size'] = math.ceil(0.05 * math.sqrt(stats['count']) if stats['count'] > 0 else 1)
            
            cluster_stats.append(stats)
        
        centroid_df = pd.DataFrame(cluster_stats)
        
        print(f"\n📊 Cluster Centroids (from Database):")
        print(centroid_df[['cluster_id', 'persona', 'PC1_mean', 'PC2_mean', 'PC3_mean', 'count', 'sphere_size']].round(3))
        
        return centroid_df
        
    except Exception as e:
        print(f"❌ Error loading from database: {e}")
        print("⚠️  Falling back to hardcoded values...")
        return calculate_cluster_centroids_fallback()
    finally:
        conn.close()

def calculate_cluster_centroids_fallback():
    """
    Fallback to hardcoded values if database doesn't have PCA scores.
    """
    
    print("🔄 Using fallback cluster centroids from successful clustering...")
    
    # From your successful clustering output
    cluster_data = {
        0: {
            'persona': 'High Value Users',
            'PC1_mean': 2.784,
            'PC2_mean': -0.203, 
            'PC3_mean': 0.442,
            'count': 502229,
            'percentage': 14.2
        },
        1: {
            'persona': 'Multi-Chain Users', 
            'PC1_mean': 9.106,
            'PC2_mean': -0.084,
            'PC3_mean': -0.555,
            'count': 131451,
            'percentage': 3.7
        },
        2: {
            'persona': 'Basic Bridge Users',
            'PC1_mean': -0.897,
            'PC2_mean': 0.039,
            'PC3_mean': -0.051,
            'count': 2893451,
            'percentage': 82.0
        }
    }
    
    cluster_stats = []
    
    for cluster_id, data in cluster_data.items():
        stats = {
            'cluster_id': cluster_id,
            'persona': data['persona'],
            'PC1_mean': data['PC1_mean'],
            'PC2_mean': data['PC2_mean'],
            'PC3_mean': data['PC3_mean'],
            'PC1_std': 0.5,  # Approximate standard deviations
            'PC2_std': 0.3,
            'PC3_std': 0.4,
            'count': data['count'],
            'percentage': data['percentage']
        }
        
        # Calculate sphere size using ceiling(0.05*sqrt(# wallets))
        stats['sphere_size'] = math.ceil(0.05 * math.sqrt(stats['count']) if stats['count'] > 0 else 1)
        
        cluster_stats.append(stats)
    
    centroid_df = pd.DataFrame(cluster_stats)
    
    print(f"\n📊 Cluster Centroids (Fallback):")
    print(centroid_df[['cluster_id', 'persona', 'PC1_mean', 'PC2_mean', 'PC3_mean', 'count', 'sphere_size']].round(3))
    
    return centroid_df

def load_sample_data_points(db_name="relay_analysis.db", sample_size=1000):
    """
    Load a random sample of individual data points from each cluster.
    """
    
    conn = sqlite3.connect(db_name)
    
    try:
        print(f"🔍 Loading random sample of {sample_size} points per cluster...")
        
        # Check if PCA columns exist
        columns_query = "PRAGMA table_info(loyalty_persona_final)"
        columns_df = pd.read_sql(columns_query, conn)
        available_columns = columns_df['name'].tolist()
        pca_columns = [col for col in available_columns if col.startswith('PC')]
        
        if len(pca_columns) < 3:
            print(f"⚠️  No PCA data available for sampling")
            return None
        
        # Sample data points from each cluster
        sample_data = []
        
        # Get distinct clusters
        cluster_query = "SELECT DISTINCT cluster_id, persona FROM loyalty_persona_final WHERE PC1 IS NOT NULL"
        clusters_df = pd.read_sql(cluster_query, conn)
        
        for _, cluster_row in clusters_df.iterrows():
            cluster_id = cluster_row['cluster_id']
            persona = cluster_row['persona']
            
            # Random sample from this cluster
            sample_query = f"""
            SELECT cluster_id, persona, PC1, PC2, PC3
            FROM loyalty_persona_final 
            WHERE cluster_id = {cluster_id} AND PC1 IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {sample_size}
            """
            
            cluster_sample = pd.read_sql(sample_query, conn)
            sample_data.append(cluster_sample)
        
        if sample_data:
            all_samples = pd.concat(sample_data, ignore_index=True)
            print(f"✅ Loaded {len(all_samples)} sample points across {len(clusters_df)} clusters")
            return all_samples
        else:
            return None
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return None
    finally:
        conn.close()

def create_3d_pca_visualization(centroid_df, output_file="pca_persona_viz.html"):
    """
    Create 3D PCA visualization showing persona cluster centroids and sample data points.
    """
    
    try:
        print("🎨 Creating 3D PCA persona visualization...")
        
        # Load sample data points
        sample_df = load_sample_data_points(sample_size=1000)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Define colors for personas (updated as requested)
        persona_colors = {
            'Basic Bridge Users': '#7e61cc',     # Relay Purple
            'High Value Users': '#8B0000',       # Optimism Red
            'Multi-Chain Users': '#00008B'       # Base Blue
        }
        
        # Add sample data points first (so centroids appear on top)
        if sample_df is not None:
            print("🔸 Adding sample data points...")
            
            for persona in sample_df['persona'].unique():
                persona_data = sample_df[sample_df['persona'] == persona]
                color = persona_colors.get(persona, '#d62728')
                
                fig.add_trace(go.Scatter3d(
                    x=persona_data['PC1'],
                    y=persona_data['PC2'], 
                    z=persona_data['PC3'],
                    mode='markers',
                    marker=dict(
                        size=2,  # Small points
                        color=color,
                        opacity=0.3,  # Semi-transparent
                        line=dict(width=0)
                    ),
                    name=f"{persona} (sample)",
                    hovertemplate=f"<b>{persona}</b><br>" +
                                 "PC1: %{x:.3f}<br>" +
                                 "PC2: %{y:.3f}<br>" +
                                 "PC3: %{z:.3f}<extra></extra>",
                    showlegend=False  # Don't clutter legend with sample points
                ))
        
        for _, row in centroid_df.iterrows():
            persona = row['persona']
            color = persona_colors.get(persona, '#d62728')  # Default red if not found
            
            # Add cluster centroid as sphere
            fig.add_trace(go.Scatter3d(
                x=[row['PC1_mean']],
                y=[row['PC2_mean']],
                z=[row['PC3_mean']],
                mode='markers',
                marker=dict(
                    size=row['sphere_size'],
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                name=f"{persona} ({row['count']:,} wallets, {row['percentage']:.1f}%)",
                hovertemplate=f"<b>{persona}</b><br>" +
                             f"Cluster ID: {row['cluster_id']}<br>" +
                             f"PC1: {row['PC1_mean']:.3f} ± {row['PC1_std']:.3f}<br>" +
                             f"PC2: {row['PC2_mean']:.3f} ± {row['PC2_std']:.3f}<br>" +
                             f"PC3: {row['PC3_mean']:.3f} ± {row['PC3_std']:.3f}<br>" +
                             f"Wallets: {row['count']:,} ({row['percentage']:.1f}%)<br>" +
                             f"Sphere Size: {row['sphere_size']}<extra></extra>"
            ))
        
        # Update layout
        title_text = "PCA Cluster Centroids by User Personas<br><sub>Sphere size = ceil(0.05×sqrt(wallet count))"
        if sample_df is not None:
            title_text += f" | {len(sample_df)} sample points shown"
        title_text += "</sub>"
        
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title="PC1 (Multi-chain/Complex Usage)",
                yaxis_title="PC2 (High Value/Specific Usage)",
                zaxis_title="PC3 (Simple Bridging Focus)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=700,
            font=dict(size=12),
            showlegend=True
        )
        
        # Save HTML file
        fig.write_html(output_file)
        print(f"✅ 3D PCA persona visualization saved to: {output_file}")
        
        # Show the plot
        fig.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

def add_pca_interpretations():
    """
    Print PCA component interpretations for reference.
    """
    
    print(f"\n🔍 PCA Component Interpretations:")
    print("=" * 60)
    print("PC1 (64.7% variance): Multi-chain/complex usage")
    print("   + Higher values = more origin chains, dest chains, distinct routes")
    print("   + Higher values = more cross-chain swaps, complex transactions")
    print("")
    print("PC2 (11.0% variance): High value/specific usage")  
    print("   + Higher values = higher transaction volumes")
    print("   + Lower values = fewer call transactions")
    print("")
    print("PC3 (8.7% variance): Simple bridging focus")
    print("   + Higher values = more bridge transactions")
    print("   + Lower values = fewer cross-chain swaps")

if __name__ == "__main__":
    print("🚀 Starting PCA Persona Visualization...")
    print("=" * 60)
    
    # Add PCA interpretations for context
    add_pca_interpretations()
    
    # Calculate cluster centroids from database (with fallback)
    centroid_df = calculate_cluster_centroids_and_stats()
    
    if centroid_df is None:
        print("❌ Cannot calculate cluster centroids")
        exit(1)
    
    # Create 3D visualization
    fig = create_3d_pca_visualization(centroid_df)
    
    if fig is not None:
        print(f"\n🎯 VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"✅ 3D PCA persona visualization created!")
        print(f"📊 Showing {len(centroid_df)} persona clusters")
        print(f"🎨 Sphere sizes based on ceil(0.05×sqrt(wallet count))")
        print(f"🔸 Sample points: 1,000 random wallets per cluster (size=2, opacity=0.3)")
        print(f"🌈 Colors: Purple (Basic), Red (High Value), Blue (Multi-Chain)")
        print(f"💾 Data source: Database (or fallback if PCA scores not saved)")
        print(f"\n💡 Interpretation:")
        print(f"   🟣 Basic Bridge Users: Low PC1, simple usage")
        print(f"   🔴 High Value Users: Moderate PC1, some complexity") 
        print(f"   🔵 Multi-Chain Users: High PC1, maximum complexity")
        print(f"\n🔍 Visualization shows:")
        print(f"   📍 Large spheres = cluster centroids")
        print(f"   🔸 Small dots = individual wallet samples")
        print(f"   📈 Spread shows cluster variance and overlap")
    else:
        print("❌ Failed to create visualization") 