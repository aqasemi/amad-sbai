#!/usr/bin/env python3
"""
Cardiac Arrest Biomarker Comparison Analysis
=============================================
Compares lab test values of cardiac arrest patients (10 years before death)
with non-cardiac death patients and the general healthy population.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

# Define data directories
BASE_DIR = Path('/home/ubuntu/datathon')
DATA_DIR = BASE_DIR / 'data2' / 'healththon - data'
DEATHS_DIR = DATA_DIR / 'Deaths'
LABS_DIR = DATA_DIR / 'LABs'
OUTPUT_DIR = BASE_DIR / 'cardiac_arrest_biomarker_comparison'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("CARDIAC ARREST BIOMARKER COMPARISON ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: IDENTIFY CARDIAC ARREST PATIENTS WITH DEATH DATES
# ============================================================================
print("\n" + "="*80)
print("STEP 1: IDENTIFYING CARDIAC ARREST PATIENTS")
print("="*80)

# Load death data
death_file = DEATHS_DIR / '20251002_Death Data Hashed.csv'
print(f"\nLoading death data...")
deaths_df = pd.read_csv(death_file, low_memory=False)
print(f"✓ Total death records: {len(deaths_df):,}")

# Identify cardiac arrest cases
cardiac_keywords = ['cardiac arrest', 'I46', 'heart failure', 'cardiac death']

def is_cardiac_arrest(row):
    """Check if death is related to cardiac arrest"""
    direct = str(row['directdeathcasueicd10']).lower() if pd.notna(row['directdeathcasueicd10']) else ''
    underlying = str(row['underlyingdeathcauseicd10']).lower() if pd.notna(row['underlyingdeathcauseicd10']) else ''
    direct_code = str(row['directdeathcauseicd10code']).lower() if pd.notna(row['directdeathcauseicd10code']) else ''
    underlying_code = str(row['underlyingdeathcauseicd10code']).lower() if pd.notna(row['underlyingdeathcauseicd10code']) else ''
    
    for keyword in cardiac_keywords:
        if (keyword in direct or keyword in underlying or 
            keyword in direct_code or keyword in underlying_code):
            return True
    return False

deaths_df['is_cardiac_arrest'] = deaths_df.apply(is_cardiac_arrest, axis=1)

# Parse death dates
deaths_df['deathdate_parsed'] = pd.to_datetime(deaths_df['deathdate'], errors='coerce')

# Separate groups
cardiac_deaths = deaths_df[deaths_df['is_cardiac_arrest'] == True].copy()
non_cardiac_deaths = deaths_df[deaths_df['is_cardiac_arrest'] == False].copy()

# Filter those with valid death dates
cardiac_deaths_with_date = cardiac_deaths[cardiac_deaths['deathdate_parsed'].notna()].copy()
non_cardiac_deaths_with_date = non_cardiac_deaths[non_cardiac_deaths['deathdate_parsed'].notna()].copy()

print(f"\n✓ Cardiac arrest deaths: {len(cardiac_deaths):,}")
print(f"✓ Cardiac arrest deaths with date: {len(cardiac_deaths_with_date):,}")
print(f"✓ Non-cardiac deaths: {len(non_cardiac_deaths):,}")
print(f"✓ Non-cardiac deaths with date: {len(non_cardiac_deaths_with_date):,}")

# Create lookup dictionaries
cardiac_patient_ids = set(cardiac_deaths_with_date['personalid'].values)
non_cardiac_patient_ids = set(non_cardiac_deaths_with_date['personalid'].values)

# Create death date lookup
cardiac_death_dates = dict(zip(cardiac_deaths_with_date['personalid'], 
                               cardiac_deaths_with_date['deathdate_parsed']))
non_cardiac_death_dates = dict(zip(non_cardiac_deaths_with_date['personalid'],
                                   non_cardiac_deaths_with_date['deathdate_parsed']))

print(f"\n✓ Cardiac arrest patient IDs: {len(cardiac_patient_ids):,}")
print(f"✓ Non-cardiac patient IDs: {len(non_cardiac_patient_ids):,}")

# ============================================================================
# STEP 2: DEFINE KEY BIOMARKERS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DEFINING KEY BIOMARKERS")
print("="*80)

# Define comprehensive biomarker list
key_biomarkers = {
    # Cardiac-specific
    'Troponin': ['Troponin'],
    'BNP': ['BNP', 'Brain Natriuretic Peptide'],
    
    # Lipid panel
    'Cholesterol': ['Cholesterol', 'Total Cholesterol'],
    'Triglycerides': ['Triglyceride'],
    'HDL': ['HDL'],
    'LDL': ['LDL'],
    
    # Metabolic
    'Glucose': ['Glucose'],
    'HbA1c': ['HbA1c', 'Hemoglobin A1c', 'Glycated hemoglobin'],
    
    # Renal function
    'Creatinine': ['Creatinine'],
    'Albumin': ['Albumin'],
    'Urea': ['Urea', 'Blood Urea'],
    
    # Electrolytes
    'Potassium': ['Potassium'],
    'Sodium': ['Sodium'],
    'Calcium': ['Calcium'],
    'Magnesium': ['Magnesium'],
    
    # Hematology
    'Hemoglobin': ['Hemoglobin', 'Hb'],
    'WBC': ['WBC', 'Leukocytes', 'White Blood Cell'],
    'RBC': ['RBC', 'Red Blood Cell', 'Erythrocyte'],
    'Platelet': ['Platelet'],
    'Lymphocytes': ['Lymphocyte', 'LYM'],
    'MCV': ['MCV', 'Mean Corpuscular Volume'],
    'RDW': ['RDW', 'Red Cell Distribution Width'],
    
    # Liver function
    'ALP': ['Alkaline phosphatase', 'ALP'],
    'ALT': ['ALT', 'Alanine aminotransferase'],
    'AST': ['AST', 'Aspartate aminotransferase'],
    'Bilirubin': ['Bilirubin'],
    
    # Inflammation
    'CRP': ['C-Reactive Protein', 'CRP'],
    'ESR': ['ESR', 'Erythrocyte Sedimentation Rate']
}

print(f"✓ Analyzing {len(key_biomarkers)} biomarkers")

def matches_biomarker(sub_test_name, keywords):
    """Check if a test name matches biomarker keywords"""
    if pd.isna(sub_test_name):
        return False
    sub_test_name_lower = str(sub_test_name).lower()
    return any(keyword.lower() in sub_test_name_lower for keyword in keywords)

# ============================================================================
# STEP 3: LOAD AND PROCESS LAB DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 3: LOADING AND PROCESSING LAB DATA")
print("="*80)

# Define lab files
lab_files = [
    '20250929_datathon_2_labs_above_50_alive.csv',
    '20251019_datathon_2_labs_0_35_all.csv',
    '20251019_datathon_2_labs_36_50_all.csv',
    '20251019_datathon_2_labs_above_50_death.csv'
]

# Initialize storage for biomarker data
biomarker_data = {
    'cardiac_arrest': {name: [] for name in key_biomarkers.keys()},
    'non_cardiac_death': {name: [] for name in key_biomarkers.keys()},
    'healthy_population': {name: [] for name in key_biomarkers.keys()}
}

print("\nProcessing lab files...")

for lab_file in lab_files:
    file_path = LABS_DIR / lab_file
    if not file_path.exists():
        print(f"  ⚠ Warning: {lab_file} not found")
        continue
    
    print(f"\n  Processing {lab_file}...")
    is_death_file = 'death' in lab_file.lower()
    
    chunk_count = 0
    total_records = 0
    cardiac_records = 0
    non_cardiac_records = 0
    healthy_records = 0
    
    # Read in chunks
    for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
        chunk_count += 1
        total_records += len(chunk)
        
        # Parse test date
        chunk['test_date_parsed'] = pd.to_datetime(chunk['order_date'], errors='coerce')
        
        # Process each biomarker
        for biomarker_name, keywords in key_biomarkers.items():
            # Find matching tests
            mask = chunk['sub_test_name'].apply(lambda x: matches_biomarker(x, keywords))
            matching_rows = chunk[mask].copy()
            
            if len(matching_rows) == 0:
                continue
            
            # Extract numeric values
            matching_rows['result_numeric'] = pd.to_numeric(matching_rows['result_value'], errors='coerce')
            matching_rows = matching_rows[matching_rows['result_numeric'].notna()]
            
            if len(matching_rows) == 0:
                continue
            
            # Categorize by patient type
            for _, row in matching_rows.iterrows():
                patient_id = row['personalid']
                test_date = row['test_date_parsed']
                value = row['result_numeric']
                
                # Check if cardiac arrest patient
                if patient_id in cardiac_patient_ids:
                    death_date = cardiac_death_dates.get(patient_id)
                    if pd.notna(death_date) and pd.notna(test_date):
                        # Check if test was within 10 years before death
                        years_before_death = (death_date - test_date).days / 365.25
                        if 0 <= years_before_death <= 10:
                            biomarker_data['cardiac_arrest'][biomarker_name].append(value)
                            cardiac_records += 1
                
                # Check if non-cardiac death patient
                elif patient_id in non_cardiac_patient_ids:
                    death_date = non_cardiac_death_dates.get(patient_id)
                    if pd.notna(death_date) and pd.notna(test_date):
                        # Check if test was within 10 years before death
                        years_before_death = (death_date - test_date).days / 365.25
                        if 0 <= years_before_death <= 10:
                            biomarker_data['non_cardiac_death'][biomarker_name].append(value)
                            non_cardiac_records += 1
                
                # Otherwise, healthy population
                else:
                    biomarker_data['healthy_population'][biomarker_name].append(value)
                    healthy_records += 1
        
        if chunk_count % 5 == 0:
            print(f"    Processed {chunk_count} chunks, {total_records:,} records...")
    
    print(f"    ✓ Total: {total_records:,} | Cardiac: {cardiac_records:,} | " + 
          f"Non-cardiac: {non_cardiac_records:,} | Healthy: {healthy_records:,}")

# ============================================================================
# STEP 4: CALCULATE STATISTICS FOR EACH BIOMARKER
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CALCULATING BIOMARKER STATISTICS")
print("="*80)

# Convert lists to arrays and calculate statistics
biomarker_stats = []

for biomarker_name in key_biomarkers.keys():
    cardiac_values = np.array(biomarker_data['cardiac_arrest'][biomarker_name])
    non_cardiac_values = np.array(biomarker_data['non_cardiac_death'][biomarker_name])
    healthy_values = np.array(biomarker_data['healthy_population'][biomarker_name])
    
    # Only process if we have data
    if len(cardiac_values) == 0 and len(non_cardiac_values) == 0 and len(healthy_values) == 0:
        continue
    
    # Calculate statistics for each group
    def calc_stats(values):
        if len(values) == 0:
            return {
                'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan,
                'q25': np.nan, 'q75': np.nan, 'min': np.nan, 'max': np.nan
            }
        
        # Remove extreme outliers (beyond 5 IQR)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 5 * iqr
            upper = q3 + 5 * iqr
            values_filtered = values[(values >= lower) & (values <= upper)]
            if len(values_filtered) == 0:
                values_filtered = values
        else:
            values_filtered = values
        
        return {
            'count': len(values),
            'mean': np.mean(values_filtered),
            'median': np.median(values_filtered),
            'std': np.std(values_filtered),
            'q25': np.percentile(values_filtered, 25),
            'q75': np.percentile(values_filtered, 75),
            'min': np.min(values_filtered),
            'max': np.max(values_filtered)
        }
    
    cardiac_stats = calc_stats(cardiac_values)
    non_cardiac_stats = calc_stats(non_cardiac_values)
    healthy_stats = calc_stats(healthy_values)
    
    # Statistical tests
    p_value_cardiac_vs_healthy = np.nan
    p_value_cardiac_vs_noncardiac = np.nan
    
    if len(cardiac_values) > 0 and len(healthy_values) > 0:
        try:
            _, p_value_cardiac_vs_healthy = stats.mannwhitneyu(cardiac_values, healthy_values, alternative='two-sided')
        except:
            pass
    
    if len(cardiac_values) > 0 and len(non_cardiac_values) > 0:
        try:
            _, p_value_cardiac_vs_noncardiac = stats.mannwhitneyu(cardiac_values, non_cardiac_values, alternative='two-sided')
        except:
            pass
    
    # Calculate differences
    diff_cardiac_healthy = cardiac_stats['mean'] - healthy_stats['mean'] if not np.isnan(cardiac_stats['mean']) and not np.isnan(healthy_stats['mean']) else np.nan
    diff_cardiac_noncardiac = cardiac_stats['mean'] - non_cardiac_stats['mean'] if not np.isnan(cardiac_stats['mean']) and not np.isnan(non_cardiac_stats['mean']) else np.nan
    
    # Percent difference
    pct_diff_cardiac_healthy = (diff_cardiac_healthy / healthy_stats['mean'] * 100) if not np.isnan(diff_cardiac_healthy) and healthy_stats['mean'] != 0 else np.nan
    pct_diff_cardiac_noncardiac = (diff_cardiac_noncardiac / non_cardiac_stats['mean'] * 100) if not np.isnan(diff_cardiac_noncardiac) and non_cardiac_stats['mean'] != 0 else np.nan
    
    biomarker_stats.append({
        'biomarker': biomarker_name,
        'cardiac_count': cardiac_stats['count'],
        'cardiac_mean': cardiac_stats['mean'],
        'cardiac_median': cardiac_stats['median'],
        'cardiac_std': cardiac_stats['std'],
        'non_cardiac_count': non_cardiac_stats['count'],
        'non_cardiac_mean': non_cardiac_stats['mean'],
        'non_cardiac_median': non_cardiac_stats['median'],
        'non_cardiac_std': non_cardiac_stats['std'],
        'healthy_count': healthy_stats['count'],
        'healthy_mean': healthy_stats['mean'],
        'healthy_median': healthy_stats['median'],
        'healthy_std': healthy_stats['std'],
        'diff_cardiac_healthy': diff_cardiac_healthy,
        'pct_diff_cardiac_healthy': pct_diff_cardiac_healthy,
        'diff_cardiac_noncardiac': diff_cardiac_noncardiac,
        'pct_diff_cardiac_noncardiac': pct_diff_cardiac_noncardiac,
        'p_value_cardiac_vs_healthy': p_value_cardiac_vs_healthy,
        'p_value_cardiac_vs_noncardiac': p_value_cardiac_vs_noncardiac,
        'cardiac_q25': cardiac_stats['q25'],
        'cardiac_q75': cardiac_stats['q75'],
        'healthy_q25': healthy_stats['q25'],
        'healthy_q75': healthy_stats['q75']
    })

# Convert to DataFrame
stats_df = pd.DataFrame(biomarker_stats)

# Sort by absolute percent difference from healthy
stats_df['abs_pct_diff'] = stats_df['pct_diff_cardiac_healthy'].abs()
stats_df = stats_df.sort_values('abs_pct_diff', ascending=False)

print(f"\n✓ Calculated statistics for {len(stats_df)} biomarkers")

# Print summary
print("\n" + "-"*80)
print("TOP 15 BIOMARKERS WITH LARGEST DIFFERENCES (Cardiac vs Healthy)")
print("-"*80)
print(f"{'Biomarker':<20} {'Cardiac':<12} {'Healthy':<12} {'Diff':<12} {'% Diff':<10} {'p-value':<10}")
print("-"*80)

for _, row in stats_df.head(15).iterrows():
    if not np.isnan(row['cardiac_mean']) and not np.isnan(row['healthy_mean']):
        sig = '***' if row['p_value_cardiac_vs_healthy'] < 0.001 else ('**' if row['p_value_cardiac_vs_healthy'] < 0.01 else ('*' if row['p_value_cardiac_vs_healthy'] < 0.05 else ''))
        print(f"{row['biomarker']:<20} {row['cardiac_mean']:>11.2f} {row['healthy_mean']:>11.2f} " +
              f"{row['diff_cardiac_healthy']:>11.2f} {row['pct_diff_cardiac_healthy']:>9.1f}% " +
              f"{row['p_value_cardiac_vs_healthy']:>9.3e} {sig}")

# Save statistics to CSV
stats_file = OUTPUT_DIR / 'biomarker_comparison_statistics.csv'
stats_df.to_csv(stats_file, index=False)
print(f"\n✓ Statistics saved to: {stats_file}")

# ============================================================================
# STEP 5: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*80)

# Filter biomarkers with sufficient data
biomarkers_to_plot = stats_df[
    (stats_df['cardiac_count'] >= 10) & 
    (stats_df['healthy_count'] >= 10)
].copy()

print(f"\nBiomarkers with sufficient data for visualization: {len(biomarkers_to_plot)}")

if len(biomarkers_to_plot) > 0:
    # Visualization 1: Top biomarkers - Mean comparison
    print("\nCreating mean comparison chart...")
    
    top_n = min(15, len(biomarkers_to_plot))
    top_biomarkers = biomarkers_to_plot.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(top_biomarkers))
    width = 0.25
    
    bars1 = ax.barh(x - width, top_biomarkers['cardiac_mean'], width, 
                    label='Cardiac Arrest (10y before death)', color='red', alpha=0.7, edgecolor='black')
    bars2 = ax.barh(x, top_biomarkers['non_cardiac_mean'], width,
                    label='Non-Cardiac Death (10y before death)', color='orange', alpha=0.7, edgecolor='black')
    bars3 = ax.barh(x + width, top_biomarkers['healthy_mean'], width,
                    label='Healthy Population', color='green', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Biomarker', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Biomarkers: Mean Value Comparison\n(Ranked by difference from healthy population)', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(top_biomarkers['biomarker'])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'biomarker_mean_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Mean comparison saved to: {output_file}")
    plt.close()
    
    # Visualization 2: Percent difference chart
    print("\nCreating percent difference chart...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort by percent difference
    sorted_data = top_biomarkers.sort_values('pct_diff_cardiac_healthy')
    
    colors = ['red' if x > 0 else 'blue' for x in sorted_data['pct_diff_cardiac_healthy']]
    bars = ax.barh(range(len(sorted_data)), sorted_data['pct_diff_cardiac_healthy'], 
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Biomarker', fontsize=12, fontweight='bold')
    ax.set_xlabel('Percent Difference from Healthy Population (%)', fontsize=12, fontweight='bold')
    ax.set_title('Biomarker Deviations in Cardiac Arrest Patients\n(Compared to Healthy Population)', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(sorted_data)))
    ax.set_yticklabels(sorted_data['biomarker'])
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.grid(alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(sorted_data.iterrows()):
        value = row['pct_diff_cardiac_healthy']
        ax.text(value, i, f' {value:.1f}%', va='center', 
               ha='left' if value > 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'biomarker_percent_difference.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Percent difference chart saved to: {output_file}")
    plt.close()
    
    # Visualization 3: Distribution comparisons for top biomarkers
    print("\nCreating distribution comparisons...")
    
    top_6 = biomarkers_to_plot.head(6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution Comparison: Top 6 Biomarkers', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_6.iterrows()):
        ax = axes[idx]
        biomarker_name = row['biomarker']
        
        cardiac_vals = np.array(biomarker_data['cardiac_arrest'][biomarker_name])
        non_cardiac_vals = np.array(biomarker_data['non_cardiac_death'][biomarker_name])
        healthy_vals = np.array(biomarker_data['healthy_population'][biomarker_name])
        
        # Filter outliers for better visualization
        def filter_outliers(values):
            if len(values) == 0:
                return values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                return values[(values >= lower) & (values <= upper)]
            return values
        
        cardiac_filtered = filter_outliers(cardiac_vals)
        non_cardiac_filtered = filter_outliers(non_cardiac_vals)
        healthy_filtered = filter_outliers(healthy_vals)
        
        # Determine common bins
        all_vals = np.concatenate([cardiac_filtered, non_cardiac_filtered, healthy_filtered])
        if len(all_vals) > 0:
            bins = np.histogram_bin_edges(all_vals, bins=30)
            
            if len(cardiac_filtered) > 0:
                ax.hist(cardiac_filtered, bins=bins, alpha=0.5, label='Cardiac', 
                       color='red', edgecolor='black', density=True)
            if len(non_cardiac_filtered) > 0:
                ax.hist(non_cardiac_filtered, bins=bins, alpha=0.5, label='Non-Cardiac',
                       color='orange', edgecolor='black', density=True)
            if len(healthy_filtered) > 0:
                ax.hist(healthy_filtered, bins=bins, alpha=0.5, label='Healthy',
                       color='green', edgecolor='black', density=True)
            
            # Add mean lines
            if len(cardiac_vals) > 0:
                ax.axvline(np.mean(cardiac_vals), color='darkred', linestyle='--', linewidth=2)
            if len(healthy_vals) > 0:
                ax.axvline(np.mean(healthy_vals), color='darkgreen', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{biomarker_name}\n({row["pct_diff_cardiac_healthy"]:+.1f}% from healthy)', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'biomarker_distributions_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution comparisons saved to: {output_file}")
    plt.close()
    
    # Visualization 4: Box plots for top biomarkers
    print("\nCreating box plot comparisons...")
    
    top_8 = biomarkers_to_plot.head(8)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Box Plot Comparison: Top 8 Biomarkers', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_8.iterrows()):
        ax = axes[idx]
        biomarker_name = row['biomarker']
        
        cardiac_vals = biomarker_data['cardiac_arrest'][biomarker_name]
        non_cardiac_vals = biomarker_data['non_cardiac_death'][biomarker_name]
        healthy_vals = biomarker_data['healthy_population'][biomarker_name]
        
        # Filter outliers
        def filter_for_boxplot(values):
            if len(values) == 0:
                return values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                return [v for v in values if lower <= v <= upper]
            return values
        
        data_to_plot = []
        labels = []
        
        if len(cardiac_vals) > 0:
            data_to_plot.append(filter_for_boxplot(cardiac_vals))
            labels.append(f'Cardiac\n(n={len(cardiac_vals)})')
        
        if len(non_cardiac_vals) > 0:
            data_to_plot.append(filter_for_boxplot(non_cardiac_vals))
            labels.append(f'Non-Cardiac\n(n={len(non_cardiac_vals)})')
        
        if len(healthy_vals) > 0:
            data_to_plot.append(filter_for_boxplot(healthy_vals))
            labels.append(f'Healthy\n(n={len(healthy_vals)})')
        
        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showfliers=False)
            
            # Color the boxes
            colors = ['red', 'orange', 'green'][:len(data_to_plot)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(f'{biomarker_name}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=7)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'biomarker_boxplots_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Box plot comparisons saved to: {output_file}")
    plt.close()
    
    # Visualization 5: Statistical significance heatmap
    print("\nCreating statistical significance visualization...")
    
    top_20 = biomarkers_to_plot.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create data for heatmap
    sig_data = []
    biomarker_names = []
    
    for _, row in top_20.iterrows():
        biomarker_names.append(row['biomarker'])
        
        # Convert p-values to significance scores
        p_healthy = row['p_value_cardiac_vs_healthy']
        p_noncardiac = row['p_value_cardiac_vs_noncardiac']
        
        # -log10(p-value) for visualization
        sig_healthy = -np.log10(p_healthy) if not np.isnan(p_healthy) and p_healthy > 0 else 0
        sig_noncardiac = -np.log10(p_noncardiac) if not np.isnan(p_noncardiac) and p_noncardiac > 0 else 0
        
        sig_data.append([sig_healthy, sig_noncardiac])
    
    sig_array = np.array(sig_data)
    
    im = ax.imshow(sig_array, cmap='Reds', aspect='auto')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Cardiac vs\nHealthy', 'Cardiac vs\nNon-Cardiac'])
    ax.set_yticks(range(len(biomarker_names)))
    ax.set_yticklabels(biomarker_names)
    ax.set_title('Statistical Significance of Differences\n(-log10(p-value))', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)
    
    # Add significance thresholds
    ax.axhline(y=-0.5, color='white', linewidth=2)
    for i in range(len(biomarker_names)):
        ax.axhline(y=i+0.5, color='white', linewidth=0.5)
    
    # Add text annotations
    for i in range(len(biomarker_names)):
        for j in range(2):
            val = sig_array[i, j]
            sig_marker = ''
            if val >= -np.log10(0.001):
                sig_marker = '***'
            elif val >= -np.log10(0.01):
                sig_marker = '**'
            elif val >= -np.log10(0.05):
                sig_marker = '*'
            
            text = ax.text(j, i, f'{val:.1f}\n{sig_marker}',
                          ha='center', va='center', color='black' if val < 3 else 'white',
                          fontsize=7)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'biomarker_significance_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Significance heatmap saved to: {output_file}")
    plt.close()

# ============================================================================
# STEP 6: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 6: GENERATING COMPREHENSIVE REPORT")
print("="*80)

report_file = OUTPUT_DIR / 'biomarker_comparison_report.txt'

with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CARDIAC ARREST BIOMARKER COMPARISON ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("ANALYSIS OVERVIEW\n")
    f.write("-"*80 + "\n")
    f.write(f"Cardiac arrest patients: {len(cardiac_patient_ids):,}\n")
    f.write(f"Non-cardiac death patients: {len(non_cardiac_patient_ids):,}\n")
    f.write(f"Time window: Lab tests within 10 years before death\n")
    f.write(f"Biomarkers analyzed: {len(key_biomarkers)}\n")
    f.write(f"Biomarkers with data: {len(stats_df)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("TOP 20 BIOMARKERS WITH SIGNIFICANT DIFFERENCES\n")
    f.write("="*80 + "\n\n")
    
    for i, (_, row) in enumerate(stats_df.head(20).iterrows(), 1):
        f.write(f"\n{i}. {row['biomarker']}\n")
        f.write("-"*60 + "\n")
        
        f.write(f"Cardiac Arrest Patients (10y before death):\n")
        f.write(f"  N = {row['cardiac_count']:,}\n")
        if not np.isnan(row['cardiac_mean']):
            f.write(f"  Mean: {row['cardiac_mean']:.2f} ± {row['cardiac_std']:.2f}\n")
            f.write(f"  Median: {row['cardiac_median']:.2f}\n")
            f.write(f"  IQR: {row['cardiac_q25']:.2f} - {row['cardiac_q75']:.2f}\n")
        
        f.write(f"\nNon-Cardiac Death Patients (10y before death):\n")
        f.write(f"  N = {row['non_cardiac_count']:,}\n")
        if not np.isnan(row['non_cardiac_mean']):
            f.write(f"  Mean: {row['non_cardiac_mean']:.2f} ± {row['non_cardiac_std']:.2f}\n")
            f.write(f"  Median: {row['non_cardiac_median']:.2f}\n")
        
        f.write(f"\nHealthy Population:\n")
        f.write(f"  N = {row['healthy_count']:,}\n")
        if not np.isnan(row['healthy_mean']):
            f.write(f"  Mean: {row['healthy_mean']:.2f} ± {row['healthy_std']:.2f}\n")
            f.write(f"  Median: {row['healthy_median']:.2f}\n")
            f.write(f"  IQR: {row['healthy_q25']:.2f} - {row['healthy_q75']:.2f}\n")
        
        f.write(f"\nComparison:\n")
        if not np.isnan(row['diff_cardiac_healthy']):
            f.write(f"  Cardiac vs Healthy: {row['diff_cardiac_healthy']:+.2f} ({row['pct_diff_cardiac_healthy']:+.1f}%)\n")
            if not np.isnan(row['p_value_cardiac_vs_healthy']):
                sig = '***' if row['p_value_cardiac_vs_healthy'] < 0.001 else ('**' if row['p_value_cardiac_vs_healthy'] < 0.01 else ('*' if row['p_value_cardiac_vs_healthy'] < 0.05 else 'ns'))
                f.write(f"  p-value: {row['p_value_cardiac_vs_healthy']:.3e} {sig}\n")
        
        if not np.isnan(row['diff_cardiac_noncardiac']):
            f.write(f"  Cardiac vs Non-Cardiac Death: {row['diff_cardiac_noncardiac']:+.2f} ({row['pct_diff_cardiac_noncardiac']:+.1f}%)\n")
            if not np.isnan(row['p_value_cardiac_vs_noncardiac']):
                sig = '***' if row['p_value_cardiac_vs_noncardiac'] < 0.001 else ('**' if row['p_value_cardiac_vs_noncardiac'] < 0.01 else ('*' if row['p_value_cardiac_vs_noncardiac'] < 0.05 else 'ns'))
                f.write(f"  p-value: {row['p_value_cardiac_vs_noncardiac']:.3e} {sig}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS AND RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    
    # Identify biomarkers with significant differences
    significant_biomarkers = stats_df[
        (stats_df['p_value_cardiac_vs_healthy'] < 0.05) &
        (stats_df['abs_pct_diff'] > 5)
    ]
    
    f.write(f"Biomarkers with significant differences (p<0.05, >5% difference): {len(significant_biomarkers)}\n\n")
    
    # Elevated biomarkers
    elevated = significant_biomarkers[significant_biomarkers['pct_diff_cardiac_healthy'] > 0].head(10)
    if len(elevated) > 0:
        f.write("Biomarkers ELEVATED in Cardiac Arrest Patients:\n")
        for _, row in elevated.iterrows():
            f.write(f"  • {row['biomarker']}: {row['pct_diff_cardiac_healthy']:+.1f}% higher\n")
        f.write("\n")
    
    # Decreased biomarkers
    decreased = significant_biomarkers[significant_biomarkers['pct_diff_cardiac_healthy'] < 0].head(10)
    if len(decreased) > 0:
        f.write("Biomarkers DECREASED in Cardiac Arrest Patients:\n")
        for _, row in decreased.iterrows():
            f.write(f"  • {row['biomarker']}: {row['pct_diff_cardiac_healthy']:.1f}% lower\n")
        f.write("\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RECOMMENDATIONS FOR PREDICTIVE MODELING\n")
    f.write("="*80 + "\n\n")
    
    f.write("Top biomarkers to include in cardiac arrest risk prediction:\n\n")
    
    top_predictive = stats_df[
        (stats_df['cardiac_count'] >= 50) &
        (stats_df['healthy_count'] >= 100) &
        (stats_df['p_value_cardiac_vs_healthy'] < 0.01)
    ].head(15)
    
    for i, (_, row) in enumerate(top_predictive.iterrows(), 1):
        f.write(f"{i:2d}. {row['biomarker']:<20} (N={row['cardiac_count']:,}, " +
               f"diff={row['pct_diff_cardiac_healthy']:+.1f}%, p={row['p_value_cardiac_vs_healthy']:.2e})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"\n✓ Comprehensive report saved to: {report_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print(f"  1. biomarker_comparison_statistics.csv")
print(f"  2. biomarker_mean_comparison.png")
print(f"  3. biomarker_percent_difference.png")
print(f"  4. biomarker_distributions_comparison.png")
print(f"  5. biomarker_boxplots_comparison.png")
print(f"  6. biomarker_significance_heatmap.png")
print(f"  7. biomarker_comparison_report.txt")
print("\n" + "="*80)
