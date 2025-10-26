"""
Exploratory Data Analysis for NHANES III Data
Run this after download_nhanes_iii.py to analyze the downloaded data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
os.makedirs('nhanes_iii_eda', exist_ok=True)

def load_data():
    """Load the downloaded NHANES III data"""
    print("=" * 80)
    print("LOADING NHANES III DATA")
    print("=" * 80)
    
    full_path = 'nhanes_iii_data/nhanes_iii_full.csv'
    
    if not os.path.exists(full_path):
        print(f"\nERROR: {full_path} not found!")
        print("Please run download_nhanes_iii.py first to download the data.")
        return None
    
    df = pd.read_csv(full_path)
    print(f"\nLoaded {len(df):,} records")
    print(f"Total columns: {len(df.columns):,}")
    
    # Load column info if available
    column_info_path = 'nhanes_iii_data/nhanes_iii_column_info.csv'
    if os.path.exists(column_info_path):
        print("\nColumn information file found - use it to explore available columns")
    
    print(f"\nFirst 20 columns:")
    for col in df.columns[:20]:
        n_available = df[col].notna().sum()
        pct = (n_available / len(df)) * 100
        print(f"  - {col:20s}: {n_available:7,} ({pct:5.1f}%)")
    
    if len(df.columns) > 20:
        print(f"\n... and {len(df.columns) - 20} more columns")
    
    return df

def basic_summary(df):
    """Print basic summary statistics"""
    print("\n" + "=" * 80)
    print("BASIC SUMMARY STATISTICS")
    print("=" * 80)
    
    # Data shape
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Missing data summary
    print("\n" + "-" * 80)
    print("MISSING DATA SUMMARY")
    print("-" * 80)
    
    missing_summary = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = (n_missing / len(df)) * 100
        n_present = len(df) - n_missing
        pct_present = ((len(df) - n_missing) / len(df)) * 100
        
        missing_summary.append({
            'Column': col,
            'Present': n_present,
            'Present %': pct_present,
            'Missing': n_missing,
            'Missing %': pct_missing
        })
    
    missing_df = pd.DataFrame(missing_summary)
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    
    print(missing_df.to_string(index=False))
    
    # Save to file
    missing_df.to_csv('nhanes_iii_eda/missing_data_summary.csv', index=False)
    print("\n✓ Saved missing data summary to nhanes_iii_eda/missing_data_summary.csv")
    
    return missing_df

def age_analysis(df):
    """Analyze age distribution"""
    # Look for age column (could be 'age', 'HSAGEIR', etc.)
    age_col = None
    for col in ['age', 'HSAGEIR', 'HSAGEU', 'AGE']:
        if col in df.columns:
            age_col = col
            break
    
    if age_col is None:
        print("\nNo age column found")
        return
    
    print(f"\nUsing age column: {age_col}")
    
    print("\n" + "=" * 80)
    print("AGE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    age_data = df[age_col].dropna()
    
    print(f"\nAge Statistics:")
    print(f"  Count: {len(age_data):,}")
    print(f"  Mean: {age_data.mean():.1f} years")
    print(f"  Median: {age_data.median():.1f} years")
    print(f"  Min: {age_data.min():.1f} years")
    print(f"  Max: {age_data.max():.1f} years")
    print(f"  Std Dev: {age_data.std():.1f} years")
    
    # Age groups
    print("\n" + "-" * 80)
    print("AGE GROUP DISTRIBUTION")
    print("-" * 80)
    
    age_groups = pd.cut(age_data, 
                        bins=[0, 2, 5, 12, 18, 30, 50, 65, 120],
                        labels=['Infant (0-2)', 'Child (2-5)', 'Child (5-12)', 
                               'Teen (12-18)', 'Young Adult (18-30)', 
                               'Adult (30-50)', 'Middle Age (50-65)', 'Senior (65+)'])
    
    age_group_counts = age_groups.value_counts().sort_index()
    
    for group, count in age_group_counts.items():
        pct = (count / len(age_data)) * 100
        print(f"  {group:20s}: {count:6,} ({pct:5.1f}%)")
    
    # Plot age distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(age_data, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Age (years)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Age Distribution - All Individuals', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by age group
    age_group_df = pd.DataFrame({'Age': age_data, 'Age Group': age_groups})
    age_group_df.boxplot(column='Age', by='Age Group', ax=axes[1])
    axes[1].set_xlabel('Age Group', fontsize=12)
    axes[1].set_ylabel('Age (years)', fontsize=12)
    axes[1].set_title('Age Distribution by Group', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove the automatic title
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('nhanes_iii_eda/age_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved age distribution plot to nhanes_iii_eda/age_distribution.png")
    plt.close()

def demographic_analysis(df):
    """Analyze demographic variables"""
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC ANALYSIS")
    print("=" * 80)
    
    # Sex distribution - look for sex columns
    sex_col = None
    for col in ['sex', 'HSSEX', 'SEX']:
        if col in df.columns:
            sex_col = col
            break
    
    if sex_col:
        print("\n" + "-" * 80)
        print(f"SEX DISTRIBUTION (using column: {sex_col})")
        print("-" * 80)
        sex_counts = df[sex_col].value_counts()
        for sex, count in sex_counts.items():
            pct = (count / df[sex_col].notna().sum()) * 100
            sex_label = "Male" if sex == 1 else "Female" if sex == 2 else f"Code {sex}"
            print(f"  {sex_label:10s}: {count:6,} ({pct:5.1f}%)")
    
    # Race/ethnicity distribution
    race_col = None
    for col in ['race_ethnicity', 'DMARETHN', 'RACE']:
        if col in df.columns:
            race_col = col
            break
    
    if race_col:
        print("\n" + "-" * 80)
        print(f"RACE/ETHNICITY DISTRIBUTION (using column: {race_col})")
        print("-" * 80)
        race_counts = df[race_col].value_counts()
        race_labels = {
            1: "Non-Hispanic White",
            2: "Non-Hispanic Black",
            3: "Mexican American",
            4: "Other"
        }
        for race, count in race_counts.items():
            pct = (count / df[race_col].notna().sum()) * 100
            race_label = race_labels.get(race, f"Code {race}")
            print(f"  {race_label:25s}: {count:6,} ({pct:5.1f}%)")

def biomarker_analysis(df):
    """Analyze biomarker distributions"""
    # Look for common lab value columns (typically start with specific prefixes)
    lab_prefixes = ['AMP', 'CEP', 'SGP', 'CRP', 'LMP', 'MVP', 'RWP', 'APP', 'WCP',
                   'TCP', 'GHP', 'BCP', 'HCP', 'MCP']
    
    lab_columns = [col for col in df.columns 
                   if any(col.startswith(prefix) for prefix in lab_prefixes)]
    
    if not lab_columns:
        print("\nNo lab/biomarker columns found in dataset")
        print("Note: With all columns loaded, specific biomarker names may vary")
        return
    
    available_biomarkers = lab_columns[:20]  # Show first 20 lab values
    
    print("\n" + "=" * 80)
    print("BIOMARKER ANALYSIS")
    print("=" * 80)
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("BIOMARKER SUMMARY STATISTICS")
    print("-" * 80)
    
    biomarker_stats = []
    for biomarker in available_biomarkers:
        data = df[biomarker].dropna()
        if len(data) > 0:
            stats = {
                'Biomarker': biomarker,
                'Count': len(data),
                'Mean': data.mean(),
                'Median': data.median(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75)
            }
            biomarker_stats.append(stats)
    
    stats_df = pd.DataFrame(biomarker_stats)
    print(stats_df.to_string(index=False))
    
    stats_df.to_csv('nhanes_iii_eda/biomarker_statistics.csv', index=False)
    print("\n✓ Saved biomarker statistics to nhanes_iii_eda/biomarker_statistics.csv")
    
    # Plot biomarker distributions
    n_biomarkers = len(available_biomarkers)
    n_cols = 3
    n_rows = (n_biomarkers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_biomarkers > 1 else [axes]
    
    for idx, biomarker in enumerate(available_biomarkers):
        data = df[biomarker].dropna()
        
        axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(biomarker.replace('_', ' ').title(), fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_title(f'{biomarker.replace("_", " ").title()}\n(n={len(data):,})', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_biomarkers), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('nhanes_iii_eda/biomarker_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved biomarker distributions to nhanes_iii_eda/biomarker_distributions.png")
    plt.close()

def complete_cases_analysis(df):
    """Analyze data completeness"""
    print("\n" + "=" * 80)
    print("DATA COMPLETENESS ANALYSIS")
    print("=" * 80)
    
    # Calculate completeness per record
    completeness_per_record = df.notna().sum(axis=1) / len(df.columns) * 100
    
    print(f"\nCompleteness per record (% of non-null columns):")
    print(f"  Mean: {completeness_per_record.mean():.1f}%")
    print(f"  Median: {completeness_per_record.median():.1f}%")
    print(f"  Min: {completeness_per_record.min():.1f}%")
    print(f"  Max: {completeness_per_record.max():.1f}%")
    
    # Records by completeness level
    print(f"\nRecords by completeness level:")
    completeness_bins = [0, 25, 50, 75, 90, 95, 100]
    completeness_labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-95%', '95-100%']
    completeness_groups = pd.cut(completeness_per_record, bins=completeness_bins, labels=completeness_labels)
    
    for label in completeness_labels:
        count = (completeness_groups == label).sum()
        pct = (count / len(df)) * 100
        print(f"  {label:10s}: {count:7,} ({pct:5.1f}%)")
    
    # Top 20 most complete columns
    print(f"\nTop 20 most complete columns:")
    column_completeness = df.notna().sum().sort_values(ascending=False)
    for i, (col, count) in enumerate(column_completeness.head(20).items(), 1):
        pct = (count / len(df)) * 100
        print(f"  {i:2d}. {col:20s}: {count:7,} ({pct:5.1f}%)")

def correlation_analysis(df):
    """Analyze correlations between numeric variables"""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Select numeric columns with reasonable completeness (>50%)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    complete_enough = [col for col in numeric_cols 
                       if df[col].notna().sum() / len(df) > 0.5]
    
    if len(complete_enough) < 2:
        print("\nNot enough complete numeric variables for correlation analysis")
        return
    
    # Limit to first 30 most complete columns for visualization
    column_completeness = df[complete_enough].notna().sum().sort_values(ascending=False)
    available_cols = column_completeness.head(30).index.tolist()
    
    print(f"\nAnalyzing correlations for top {len(available_cols)} most complete numeric columns")
    
    # Calculate correlations
    corr_data = df[available_cols].dropna()
    
    if len(corr_data) == 0:
        print("\nNo complete cases for correlation analysis")
        return
    
    print(f"Calculating correlations on {len(corr_data):,} complete records")
    corr_matrix = corr_data.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
    plt.title(f'Correlation Heatmap (Top {len(available_cols)} Numeric Variables)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('nhanes_iii_eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved correlation heatmap to nhanes_iii_eda/correlation_heatmap.png")
    plt.close()
    
    # Find and print strongest correlations (excluding diagonal)
    print("\nStrongest correlations (absolute value > 0.5):")
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    for var1, var2, corr in strong_corrs[:20]:  # Show top 20
        print(f"  {var1:15s} <-> {var2:15s}: {corr:6.3f}")

def generate_summary_report(df, missing_df):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("NHANES III DATA - EXPLORATORY DATA ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Dataset overview
    report.append("DATASET OVERVIEW")
    report.append("-" * 80)
    report.append(f"Total records: {len(df):,}")
    report.append(f"Total columns: {len(df.columns):,}")
    report.append("")
    
    # Age distribution
    age_col = None
    for col in ['age', 'HSAGEIR', 'HSAGEU', 'AGE']:
        if col in df.columns:
            age_col = col
            break
    
    if age_col:
        age_data = df[age_col].dropna()
        report.append(f"AGE DISTRIBUTION (using column: {age_col})")
        report.append("-" * 80)
        report.append(f"Records with age: {len(age_data):,}")
        report.append(f"Mean age: {age_data.mean():.1f} years")
        report.append(f"Median age: {age_data.median():.1f} years")
        report.append(f"Age range: {age_data.min():.1f} - {age_data.max():.1f} years")
        report.append("")
    
    # Data completeness
    report.append("DATA COMPLETENESS")
    report.append("-" * 80)
    completeness_per_record = df.notna().sum(axis=1) / len(df.columns) * 100
    report.append(f"Mean completeness per record: {completeness_per_record.mean():.1f}%")
    report.append(f"Median completeness per record: {completeness_per_record.median():.1f}%")
    report.append("")
    
    # Top 10 most complete columns
    report.append("TOP 10 MOST COMPLETE COLUMNS")
    report.append("-" * 80)
    column_completeness = df.notna().sum().sort_values(ascending=False)
    for i, (col, count) in enumerate(column_completeness.head(10).items(), 1):
        pct = (count / len(df)) * 100
        report.append(f"{i:2d}. {col:20s}: {count:7,} ({pct:5.1f}%)")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open('nhanes_iii_eda/eda_summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n✓ Saved summary report to nhanes_iii_eda/eda_summary_report.txt")
    print("\n" + report_text)

def main():
    """Main EDA function"""
    print("\n" + "=" * 80)
    print("NHANES III EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print("\nThis script performs comprehensive EDA on downloaded NHANES III data")
    print("Make sure you've run download_nhanes_iii.py first!")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run all analyses
    missing_df = basic_summary(df)
    age_analysis(df)
    demographic_analysis(df)
    biomarker_analysis(df)
    complete_cases_analysis(df)
    correlation_analysis(df)
    generate_summary_report(df, missing_df)
    
    print("\n" + "=" * 80)
    print("✓ EDA COMPLETE!")
    print("=" * 80)
    print("\nAll results saved in: nhanes_iii_eda/")
    print("\nFiles created:")
    print("  - missing_data_summary.csv")
    print("  - age_distribution.png")
    print("  - biomarker_statistics.csv")
    print("  - biomarker_distributions.png")
    print("  - correlation_heatmap.png")
    print("  - eda_summary_report.txt")

if __name__ == "__main__":
    main()
