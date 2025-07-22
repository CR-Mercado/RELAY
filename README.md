# Relay Bridge User Analysis

Analysis of Relay bridge users from January 1 - June 30, 2025, focused on user segmentation, loyalty patterns, and competitive positioning against Across-v3 and DeBridge.

## Data Sources

- **Relay transactions**: Processed from `relay_*.csv` files 
- **Competitive data**: Sourced from Flipside Crypto via `wallet_source_platform_dest_amount_2025_dump.csv`
- **Database**: All analysis stored in SQLite (`relay_analysis.db`)

⚠️ **Note**: CSV files and the SQLite database are excluded from git and must be generated locally.

## Setup

```bash
uv sync
```

## Analysis Pipeline

Run the numbered scripts sequentially to reproduce the analysis:

### Data Processing & Exploration
- **`01_exploration.py`**: Initial data exploration of relay CSV files, creates combined sample for schema validation.
- **`02_makesqlite.py`**: Creates SQLite database (`relay_analysis.db`) from all relay CSV files with standardized schema.
- **`03_topwallets.py`**: Identifies top wallets by transaction count for outlier analysis and exclusion planning.
- **`04_the10kclub.py`**: Analyzes wallets with >10K transactions to identify programmatic usage patterns requiring exclusion.

### Data Cleaning & Feature Engineering  
- **`05_exclusions_applied.py`**: Creates cleaned dataset excluding LiFi Diamond contracts, burn addresses, and other programmatic accounts.
- **`06_wallet_aggregation_exploration.py`**: Generates wallet-level feature aggregations and summary statistics for clustering preparation.

### User Segmentation
- **`07_wallet_hierarchical_clustering.py`**: Performs PCA and K-means clustering (k=3) to identify user personas based on transaction patterns.
- **`08_first_usage_chains.py`**: Analyzes first-time usage patterns by source/destination chain pairs to understand onboarding flows.

### Competitive & Loyalty Analysis
- **`09_reference_window_across_debridge_users.py`**: Loads competitive platform data (Across-v3, DeBridge) from Flipside dump for loyalty analysis.
- **`10_loyalty_aggregate.py`**: Combines Relay users with competitive usage to create loyalty segments (relay-only vs multi-platform).
- **`11_loyalty_persona.py`**: Performs persona analysis specifically on the loyalty-filtered user set with competitive context.

### Results & Visualization
- **`12_persona_viz.py`**: Generates PCA visualization plots showing the three-cluster persona solution.
- **`13_persona_averages.py`**: Calculates and exports persona characteristic averages for reporting tables.
- **`14_loyalty_characteristics.py`**: Analyzes marginal differences between loyal vs multi-platform users within each persona cluster.

## Key Outputs

### User Personas (3 clusters)
- **🟣 Basic Bridge Users**: Simple, low-frequency users seeking specific routes
- **🔴 High Value Users**: Core users with the highest average transaction values but focused on a narrow range of major chains  
- **🔵 Multi-Chain Users**: Heavy users of advanced features across many chains, potentially arbitrage players.

### Loyalty Insights
- **91.6%** of users are Relay-only (3.23M wallets)
- **8.4%** are multi-platform users (295K wallets)
- Multi-platform users show significantly higher activity except for function calls (IS_CALL feature)

## External Dependencies

The analysis requires `wallet_source_platform_dest_amount_2025_dump.csv` from Flipside Crypto. The SQL query is provided in `wallet_source_platform_dest_amount_2025_dump.sql` but requires separate Flipside access to generate.

## Key Files Generated

- `relay_analysis.db` - SQLite database with all processed data
- `*_characteristics.csv` - Persona and loyalty summary tables
- `pca_persona_viz.html` - Interactive persona visualization
- `final_report.md` - Analysis summary and insights

## R Markdown Report

The analysis is synthesized in `relay_report.Rmd` which generates an HTML report with interactive tables and visualizations using the CSV outputs from the Python pipeline.
