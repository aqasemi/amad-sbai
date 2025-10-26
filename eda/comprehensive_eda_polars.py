"""
Comprehensive Exploratory Data Analysis (EDA) for Healthcare Datathon
Team AMAD - Biological Age & Chronic Disease Prediction

This script performs a thorough EDA on 5 main datasets using Polars for memory efficiency:
1. Individuals Data
2. Death Data
3. Steps Data
4. Medications Data
5. Lab Test Data
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import gc  # For garbage collection to free memory

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# MEMORY OPTIMIZATION CONFIGURATION
# ============================================================================
# Set chunk size for processing large files
CHUNK_SIZE = 100000  # Process 100k rows at a time (Polars is more efficient)

# Optional: Set to True to sample data for faster processing/testing
USE_SAMPLING = False  # Set to True if you want to work with a sample
SAMPLE_FRACTION = 0.1  # Use 10% of data if sampling enabled

# Create output directories
os.makedirs('eda_outputs', exist_ok=True)
os.makedirs('eda_outputs/plots', exist_ok=True)

print("="*80)
print("HEALTHCARE DATATHON - COMPREHENSIVE EDA (POLARS VERSION)")
print("Team AMAD: Biological Age & Chronic Disease Prediction")
print("="*80)

# ============================================================================
# SECTION 1: INITIAL DATA LOADING & VALIDATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING & VALIDATION")
print("="*80)

# Load datasets - files are in data2/ directory
print("\n[1.1] Loading all datasets...")

# Load individuals data from multiple age group files
print("Loading individuals data from multiple age groups...")
individuals_files = [
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_0_12_all.csv',
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_13_20_all.csv',
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_21_25_all.csv',
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_26_30_all.csv',
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_31_35_all.csv',
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_36_42_all.csv',
    'data2/healththon - data/Individuals/20251019_datathon_2_individuals_43_50_all.csv',
    'data2/healththon - data/Individuals/20250929_datathon_2_individuals_above_50_alive.csv'
]

individuals_dfs = []
for file in individuals_files:
    print(f"  Loading {file.split('/')[-1]}...")
    df = pl.read_csv(file, low_memory=True)
    individuals_dfs.append(df)
    print(f"    {len(df):,} records")

individuals_df = pl.concat(individuals_dfs)
print(f"  Total loaded: {len(individuals_df):,} individuals")
del individuals_dfs  # Free memory
gc.collect()

print("Loading death data...")
death_df = pl.read_csv('data2/healththon - data/Deaths/20251002_Death Data Hashed.csv', low_memory=True)
print(f"  Loaded {len(death_df):,} death records")

print("Loading steps data...")
steps_files = [
    'data2/healththon - data/Steps/20251019_datathon_2_steps_0_50_all.csv',
    'data2/healththon - data/Steps/20250930_datathon_2_steps_above_50_alive.csv'
]
steps_dfs = []
for file in steps_files:
    print(f"  Loading {file.split('/')[-1]}...")
    df = pl.read_csv(file, low_memory=True)
    steps_dfs.append(df)
steps_df = pl.concat(steps_dfs)
print(f"  Total loaded: {len(steps_df):,} steps records")
del steps_dfs  # Free memory
gc.collect()

print("Loading medications data...")
medications_files = [
    'data2/healththon - data/Medications/20251019_datathon_2_Medications_0_50_all.csv',
    'data2/healththon - data/Medications/20250930_datathon_2_Medications_above_50_alive.csv',
    'data2/healththon - data/Medications/20251019_datathon_2_Medications_above_50_death.csv'
]
medications_dfs = []
for file in medications_files:
    print(f"  Loading {file.split('/')[-1]}...")
    df = pl.read_csv(file, low_memory=True)
    medications_dfs.append(df)
medications_df = pl.concat(medications_dfs)
print(f"  Total loaded: {len(medications_df):,} medication records")
del medications_dfs  # Free memory
gc.collect()

print("Loading lab tests data...")
lab_files = [
    'data2/healththon - data/LABs/20251019_datathon_2_labs_0_35_all.csv',
    'data2/healththon - data/LABs/20251019_datathon_2_labs_36_50_all.csv',
    'data2/healththon - data/LABs/20250929_datathon_2_labs_above_50_alive.csv',
    'data2/healththon - data/LABs/20251019_datathon_2_labs_above_50_death.csv'
]
lab_dfs = []
for file in lab_files:
    print(f"  Loading {file.split('/')[-1]}...")
    df = pl.read_csv(file, low_memory=True)
    lab_dfs.append(df)
lab_test_df = pl.concat(lab_dfs)
print(f"  Total loaded: {len(lab_test_df):,} lab test records")
del lab_dfs  # Free memory
gc.collect()

print("\n✓ All datasets loaded successfully!")

# Optional sampling for faster processing
if USE_SAMPLING:
    print(f"\n[1.1.1] Sampling data for faster processing ({SAMPLE_FRACTION*100:.1f}% of data)...")
    individuals_df = individuals_df.sample(fraction=SAMPLE_FRACTION, seed=42)
    death_df = death_df.sample(fraction=SAMPLE_FRACTION, seed=42)
    steps_df = steps_df.sample(fraction=SAMPLE_FRACTION, seed=42)
    medications_df = medications_df.sample(fraction=SAMPLE_FRACTION, seed=42)
    lab_test_df = lab_test_df.sample(fraction=SAMPLE_FRACTION, seed=42)
    print("  Sampling completed!")

# Display basic information for each dataset
datasets = {
    'Individuals': individuals_df,
    'Death': death_df,
    'Steps': steps_df,
    'Medications': medications_df,
    'Lab Tests': lab_test_df
}

print("\n[1.2] Dataset Overview:")
print("-" * 80)
for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory Usage: {df.estimated_size('mb'):.2f} MB")
    print(f"\n  Column Names: {', '.join(df.columns[:10])}")
    if len(df.columns) > 10:
        print(f"  ... and {len(df.columns) - 10} more columns")
    print("\n" + "-" * 80)

# Check for missing values
print("\n[1.3] Missing Values Analysis:")
print("-" * 80)
for name, df in datasets.items():
    print(f"\n{name} Dataset - Missing Values:")
    null_counts = df.null_count()
    if null_counts.sum_horizontal().item() > 0:
        print(null_counts)
    else:
        print("  No missing values found!")

# Verify personalid integrity
print("\n[1.4] PersonalID Integrity Check:")
print("-" * 80)
print(f"Unique individuals in Individuals dataset: {individuals_df['personalid'].n_unique():,}")
print(f"Unique individuals in Death dataset: {death_df['personalid'].n_unique():,}")
print(f"Unique individuals in Steps dataset: {steps_df['personalid'].n_unique():,}")
print(f"Unique individuals in Medications dataset: {medications_df['personalid'].n_unique():,}")
print(f"Unique individuals in Lab Tests dataset: {lab_test_df['personalid'].n_unique():,}")

# Check overlaps
deceased_ids = set(death_df['personalid'].unique().to_list())
living_ids = set(individuals_df.filter(pl.col('is_dead') == 0)['personalid'].unique().to_list())
all_individuals = set(individuals_df['personalid'].unique().to_list())

print(f"\nTotal unique individuals in main dataset: {len(all_individuals):,}")
print(f"Deceased individuals: {len(deceased_ids):,}")
print(f"Living individuals: {len(living_ids):,}")
print(f"Overlap between death and individuals dataset: {len(deceased_ids & all_individuals):,}")

# ============================================================================
# SECTION 2: UNIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: UNIVARIATE ANALYSIS")
print("="*80)

# 2.1 Individuals Dataset - Age Distribution
print("\n[2.1] Age Distribution Analysis:")
print("-" * 80)
age_stats = individuals_df['age'].describe()
print(f"Age Statistics:")
print(age_stats)

# Convert to pandas for plotting (Polars doesn't have matplotlib integration yet)
individuals_pandas = individuals_df.to_pandas()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(individuals_pandas['age'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(40, color='red', linestyle='--', label='Target Min Age (40)')
axes[0].axvline(55, color='red', linestyle='--', label='Target Max Age (55)')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Age Distribution - All Individuals')
axes[0].legend()

target_age_group = individuals_pandas[(individuals_pandas['age'] >= 40) & (individuals_pandas['age'] <= 55)]
print(f"\nIndividuals in target age group (40-55): {len(target_age_group):,} ({len(target_age_group)/len(individuals_pandas)*100:.2f}%)")

axes[1].boxplot(individuals_pandas['age'].dropna())
axes[1].set_ylabel('Age')
axes[1].set_title('Age Distribution - Boxplot')
plt.tight_layout()
plt.savefig('eda_outputs/plots/01_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Identify age anomalies
print(f"Infants (age < 1): {len(individuals_pandas[individuals_pandas['age'] < 1]):,}")
print(f"Children (age < 18): {len(individuals_pandas[individuals_pandas['age'] < 18]):,}")
print(f"Elderly (age >= 80): {len(individuals_pandas[individuals_pandas['age'] >= 80]):,}")

# 2.2 BMI, Systolic, Diastolic distributions
print("\n[2.2] Clinical Measurements Distribution:")
print("-" * 80)

clinical_vars = ['bmi', 'systolic', 'diastolic', 'height', 'weight']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, var in enumerate(clinical_vars):
    if var in individuals_pandas.columns:
        data = individuals_pandas[var].dropna()
        print(f"\n{var.upper()} Statistics:")
        print(data.describe())
        
        # Identify outliers using IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        print(f"Potential outliers: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
        
        axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(var)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{var.upper()} Distribution')

axes[5].hist(individuals_pandas['age'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[5].set_xlabel('Age')
axes[5].set_ylabel('Frequency')
axes[5].set_title('Age Distribution')

plt.tight_layout()
plt.savefig('eda_outputs/plots/02_clinical_measurements.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.3 Categorical Variables
print("\n[2.3] Categorical Variables Analysis:")
print("-" * 80)

categorical_vars = ['gender', 'region_en', 'is_dead']
chronic_conditions = ['has_diabetes', 'has_hypertension', 'has_obesity', 'has_dyslipidemia']

# Gender distribution
print("\nGender Distribution:")
print(individuals_df['gender'].value_counts().sort('count', descending=True))

# Region distribution
print("\nRegion Distribution:")
print(individuals_df['region_en'].value_counts().sort('count', descending=True))

# Death status
print("\nDeath Status:")
print(individuals_df['is_dead'].value_counts().sort('count', descending=True))

# Chronic conditions
print("\nChronic Conditions Prevalence:")
for condition in chronic_conditions:
    if condition in individuals_df.columns:
        count = individuals_df[condition].sum()
        pct = (count / len(individuals_df)) * 100
        print(f"  {condition}: {count:,} ({pct:.2f}%)")

# Plot categorical variables
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Gender
gender_counts = individuals_df['gender'].value_counts().sort('count', descending=True).to_pandas()
gender_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Gender Distribution')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Region (top 10)
region_counts = individuals_df['region_en'].value_counts().sort('count', descending=True).head(10).to_pandas()
region_counts.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Top 10 Regions')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

# Death status
death_counts = individuals_df['is_dead'].value_counts().sort('count', descending=True).to_pandas()
death_counts.plot(kind='bar', ax=axes[2], color='crimson')
axes[2].set_title('Death Status')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=0)

# Chronic conditions
for idx, condition in enumerate(chronic_conditions):
    if condition in individuals_df.columns and idx + 3 < len(axes):
        condition_counts = individuals_df[condition].value_counts().sort('count', descending=True).to_pandas()
        condition_counts.plot(kind='bar', ax=axes[idx+3], color='darkgreen')
        axes[idx+3].set_title(condition.replace('has_', '').replace('_', ' ').title())
        axes[idx+3].set_ylabel('Count')
        axes[idx+3].tick_params(axis='x', rotation=0)

# Hide any unused subplots
for idx in range(len(chronic_conditions) + 3, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('eda_outputs/plots/03_categorical_variables.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.4 Medications Analysis
print("\n[2.4] Medications Analysis:")
print("-" * 80)
print(f"Total medication records: {len(medications_df):,}")
print(f"Unique drugs: {medications_df['drug_name'].n_unique():,}")
print(f"Unique individuals with medications: {medications_df['personalid'].n_unique():,}")

print("\nTop 20 Most Prescribed Medications:")
top_meds = medications_df['drug_name'].value_counts().sort('count', descending=True).head(20)
print(top_meds)

fig, ax = plt.subplots(figsize=(12, 8))
top_meds_pandas = top_meds.to_pandas()
top_meds_pandas.plot(kind='barh', ax=ax, color='teal')
ax.set_xlabel('Number of Prescriptions')
ax.set_title('Top 20 Most Prescribed Medications')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('eda_outputs/plots/04_top_medications.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.5 Lab Tests Analysis
print("\n[2.5] Lab Tests Analysis:")
print("-" * 80)
print(f"Total lab test records: {len(lab_test_df):,}")
print(f"Unique test types: {lab_test_df['test_name'].n_unique():,}")
print(f"Unique individuals with lab tests: {lab_test_df['personalid'].n_unique():,}")

print("\nTop 20 Most Common Lab Tests:")
top_tests = lab_test_df['test_name'].value_counts().sort('count', descending=True).head(20)
print(top_tests)

fig, ax = plt.subplots(figsize=(12, 8))
top_tests_pandas = top_tests.to_pandas()
top_tests_pandas.plot(kind='barh', ax=ax, color='purple')
ax.set_xlabel('Number of Tests')
ax.set_title('Top 20 Most Common Lab Tests')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('eda_outputs/plots/05_top_lab_tests.png', dpi=300, bbox_inches='tight')
plt.close()

# Check for unit standardization issues
print("\nLab Test Unit Analysis (for key tests):")
key_tests = ['glucose', 'cholesterol', 'hba1c', 'hemoglobin', 'creatinine']
for test in key_tests:
    test_data = lab_test_df.filter(pl.col('test_name').str.contains(test, literal=False))
    if len(test_data) > 0:
        print(f"\n{test.upper()} - Units used:")
        print(test_data['test_unit'].value_counts().sort('count', descending=True))

# Analyze result values for common test
if len(lab_test_df) > 0:
    most_common_test = lab_test_df['test_name'].value_counts().sort('count', descending=True).head(1)['test_name'].item()
    test_data = lab_test_df.filter(pl.col('test_name') == most_common_test)
    
    # Convert result_value to numeric
    test_data = test_data.with_columns(
        pl.col('result_value').str.extract(r'(\d+\.?\d*)').cast(pl.Float64).alias('result_value_numeric')
    )
    
    print(f"\nResult Value Distribution for '{most_common_test}':")
    print(test_data['result_value_numeric'].describe())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    test_data_pandas = test_data.to_pandas()
    test_data_pandas['result_value_numeric'].dropna().hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Result Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Result Value Distribution - {most_common_test}')
    plt.tight_layout()
    plt.savefig('eda_outputs/plots/06_common_test_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2.6 Death Data Analysis
print("\n[2.6] Death Data Analysis:")
print("-" * 80)
print(f"Total death records: {len(death_df):,}")

# Check for death cause columns
death_cause_col = None
if 'directdeathcasueicd10' in death_df.columns:
    death_cause_col = 'directdeathcasueicd10'
elif 'Direct_death_cause_des' in death_df.columns:
    death_cause_col = 'Direct_death_cause_des'

if death_cause_col and death_df[death_cause_col].null_count().item() < len(death_df):
    print(f"\nTop 20 Direct Causes of Death (using {death_cause_col}):")
    top_causes = death_df[death_cause_col].value_counts().sort('count', descending=True).head(20)
    print(top_causes)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_causes_pandas = top_causes.to_pandas()
    top_causes_pandas.plot(kind='barh', ax=ax, color='darkred')
    ax.set_xlabel('Number of Deaths')
    ax.set_title('Top 20 Direct Causes of Death')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('eda_outputs/plots/07_death_causes.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("\nDeath cause information not available or has too many missing values.")

# ============================================================================
# SECTION 3: BIVARIATE & MULTIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: BIVARIATE & MULTIVARIATE ANALYSIS")
print("="*80)

# 3.1 Correlation Analysis
print("\n[3.1] Correlation Analysis:")
print("-" * 80)

numeric_cols = ['age', 'bmi', 'height', 'weight', 'systolic', 'diastolic', 
                'total_outpatient_visits', 'total_inpatient_visits', 'total_emergency_visits']
numeric_cols = [col for col in numeric_cols if col in individuals_df.columns]

# Convert to pandas for correlation analysis
correlation_df = individuals_pandas[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_df)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap - Clinical & Visit Metrics')
plt.tight_layout()
plt.savefig('eda_outputs/plots/08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.2 BMI and Age by Diabetes Status
print("\n[3.2] Clinical Measurements by Chronic Conditions:")
print("-" * 80)

if 'has_diabetes' in individuals_df.columns:
    print("\nBMI by Diabetes Status:")
    bmi_by_diabetes = individuals_df.group_by('has_diabetes').agg(pl.col('bmi').describe()).sort('has_diabetes')
    print(bmi_by_diabetes)
    
    print("\nAge by Diabetes Status:")
    age_by_diabetes = individuals_df.group_by('has_diabetes').agg(pl.col('age').describe()).sort('has_diabetes')
    print(age_by_diabetes)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # BMI by diabetes
    individuals_pandas.boxplot(column='bmi', by='has_diabetes', ax=axes[0, 0])
    axes[0, 0].set_title('BMI by Diabetes Status')
    axes[0, 0].set_xlabel('Has Diabetes')
    axes[0, 0].set_ylabel('BMI')
    plt.sca(axes[0, 0])
    plt.xticks([1, 2], ['No', 'Yes'])
    
    # Age by diabetes
    individuals_pandas.boxplot(column='age', by='has_diabetes', ax=axes[0, 1])
    axes[0, 1].set_title('Age by Diabetes Status')
    axes[0, 1].set_xlabel('Has Diabetes')
    axes[0, 1].set_ylabel('Age')
    plt.sca(axes[0, 1])
    plt.xticks([1, 2], ['No', 'Yes'])
    
    # BMI by hypertension
    if 'has_hypertension' in individuals_df.columns:
        individuals_pandas.boxplot(column='bmi', by='has_hypertension', ax=axes[1, 0])
        axes[1, 0].set_title('BMI by Hypertension Status')
        axes[1, 0].set_xlabel('Has Hypertension')
        axes[1, 0].set_ylabel('BMI')
        plt.sca(axes[1, 0])
        plt.xticks([1, 2], ['No', 'Yes'])
        
        # Age by hypertension
        individuals_pandas.boxplot(column='age', by='has_hypertension', ax=axes[1, 1])
        axes[1, 1].set_title('Age by Hypertension Status')
        axes[1, 1].set_xlabel('Has Hypertension')
        axes[1, 1].set_ylabel('Age')
        plt.sca(axes[1, 1])
        plt.xticks([1, 2], ['No', 'Yes'])
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('eda_outputs/plots/09_conditions_vs_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3.3 Living vs Deceased Comparison
print("\n[3.3] Living vs Deceased Population Comparison:")
print("-" * 80)

living_age = individuals_df.filter(pl.col('is_dead') == 0)['age'].mean()
deceased_age = individuals_df.filter(pl.col('is_dead') == 1)['age'].mean()

print("\nAverage Age:")
print(f"Living: {living_age:.2f}")
print(f"Deceased: {deceased_age:.2f}")

living_bmi = individuals_df.filter(pl.col('is_dead') == 0)['bmi'].mean()
deceased_bmi = individuals_df.filter(pl.col('is_dead') == 1)['bmi'].mean()

print("\nAverage BMI:")
print(f"Living: {living_bmi:.2f}")
print(f"Deceased: {deceased_bmi:.2f}")

# 3.4 Regional Analysis
print("\n[3.4] Regional Analysis of Chronic Conditions:")
print("-" * 80)

if 'region_en' in individuals_df.columns:
    top_regions = individuals_df['region_en'].value_counts().sort('count', descending=True).head(5)['region_en'].to_list()
    region_data = individuals_df.filter(pl.col('region_en').is_in(top_regions))
    
    for condition in chronic_conditions:
        if condition in region_data.columns:
            print(f"\n{condition.replace('has_', '').title()} Prevalence by Top 5 Regions:")
            condition_by_region = region_data.group_by('region_en').agg(
                (pl.col(condition).mean() * 100).alias('prevalence_pct')
            ).sort('prevalence_pct', descending=True)
            print(condition_by_region)

# ============================================================================
# SECTION 4: DATA QUALITY & CLEANING RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: DATA QUALITY & CLEANING RECOMMENDATIONS")
print("="*80)

data_quality_report = []

# Missing values issues
print("\n[4.1] Missing Data Issues:")
print("-" * 80)
for name, df in datasets.items():
    null_counts = df.null_count()
    total_rows = len(df)
    critical_missing = []
    
    for col in df.columns:
        null_count = null_counts[col].item()
        pct = (null_count / total_rows) * 100
        if pct > 30:
            critical_missing.append((col, pct))
    
    if critical_missing:
        print(f"\n{name} - Columns with >30% missing data:")
        for col, pct in critical_missing:
            print(f"  - {col}: {pct:.2f}%")
            data_quality_report.append(f"{name}: {col} has {pct:.2f}% missing values")

# Outliers
print("\n[4.2] Outlier Detection:")
print("-" * 80)
for var in ['bmi', 'systolic', 'diastolic', 'age']:
    if var in individuals_df.columns:
        data = individuals_df[var].drop_nulls()
        Q1 = data.quantile(0.25).item()
        Q3 = data.quantile(0.75).item()
        IQR = Q3 - Q1
        outliers = data.filter((pl.col(var) < Q1 - 1.5*IQR) | (pl.col(var) > Q3 + 1.5*IQR))
        if len(outliers) > 0:
            print(f"{var.upper()}: {len(outliers)} outliers ({len(outliers)/len(data)*100:.2f}%)")
            print(f"  Range: {data.min().item():.2f} - {data.max().item():.2f}")
            print(f"  Outlier range: <{Q1 - 1.5*IQR:.2f} or >{Q3 + 1.5*IQR:.2f}")
            data_quality_report.append(f"{var} has {len(outliers)} outliers that need review")

# Unit standardization
print("\n[4.3] Unit Standardization Issues:")
print("-" * 80)
for test in ['glucose', 'cholesterol']:
    test_data = lab_test_df.filter(pl.col('test_name').str.contains(test, literal=False))
    if len(test_data) > 0:
        unique_units = test_data['test_unit'].n_unique()
        if unique_units > 1:
            print(f"{test.upper()}: {unique_units} different units detected")
            print(f"  Units: {test_data['test_unit'].unique().to_list()}")
            data_quality_report.append(f"{test} tests have {unique_units} different units requiring standardization")

# Summary recommendations
print("\n[4.4] Cleaning Recommendations:")
print("-" * 80)
print("""
1. MISSING DATA HANDLING:
   - For clinical measurements (BMI, blood pressure): Consider imputation using median by age group
   - For categorical variables: Create 'Unknown' category if >5% missing
   - For visit counts: Missing likely means 0 visits - consider filling with 0

2. OUTLIER TREATMENT:
   - BMI: Values <10 or >80 should be reviewed (likely data entry errors)
   - Blood pressure: Systolic <70 or >250, Diastolic <40 or >150 need review
   - Age: Verify records with age <1 or >110
   - Consider capping outliers at 99th percentile for modeling

3. UNIT STANDARDIZATION:
   - Lab tests: Create conversion functions for common tests (glucose, cholesterol)
   - Standardize to most common unit for each test type
   - Document all conversions for reproducibility

4. DATA TYPE CONVERSIONS:
   - Convert date columns (prescription_time, order_date, deathdate) to datetime
   - Ensure boolean flags are 0/1 integers for modeling
   - Normalize text fields (drug names, test names) for consistency

5. FEATURE ENGINEERING OPPORTUNITIES:
   - Create age groups (esp. highlighting 40-55 target range)
   - Calculate time-based features from medication/lab test timestamps
   - Aggregate steps data by year to create activity profiles
   - Create comorbidity count (sum of chronic conditions)
   - Calculate medication/lab test frequency per individual
""")

# ============================================================================
# SECTION 5: CREATE COMBINED DATASET
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: CREATING COMBINED DATASET")
print("="*80)

print("\n[5.1] Aggregating transactional data...")

# Aggregate medications
print("Aggregating medications data...")
med_agg = medications_df.group_by('personalid').agg([
    pl.col('drug_name').count().alias('total_prescriptions'),
    pl.col('drug_code').n_unique().alias('unique_drugs')
])

# Aggregate lab tests
print("Aggregating lab tests data...")
lab_agg = lab_test_df.group_by('personalid').agg([
    pl.col('test_name').count().alias('total_lab_tests'),
    pl.col('test_code').n_unique().alias('unique_test_types')
])

# Aggregate steps data (latest year)
print("Aggregating steps data...")
steps_latest = steps_df.sort('year', descending=True).group_by('personalid').first()
steps_agg = steps_latest.select([
    'personalid', 'steps', 'calories', 'distance', 'movetimeduration'
]).rename({
    'steps': 'latest_steps',
    'calories': 'latest_calories', 
    'distance': 'latest_distance',
    'movetimeduration': 'latest_movetimeduration'
})

# Merge death data
print("Merging death data...")
# Check which death cause columns exist
death_cols = ['personalid', 'deathdate']
if 'directdeathcauseicd10code' in death_df.columns:
    death_cols.append('directdeathcauseicd10code')
if 'directdeathcasueicd10' in death_df.columns:
    death_cols.append('directdeathcasueicd10')
if 'underlyingdeathcauseicd10code' in death_df.columns:
    death_cols.append('underlyingdeathcauseicd10code')
if 'underlyingdeathcauseicd10' in death_df.columns:
    death_cols.append('underlyingdeathcauseicd10')

death_subset = death_df.select(death_cols)

# Start with individuals data
print("Creating master dataset...")
master_df = individuals_df.clone()

# Merge all aggregated data
master_df = master_df.join(med_agg, on='personalid', how='left')
master_df = master_df.join(lab_agg, on='personalid', how='left')
master_df = master_df.join(steps_agg, on='personalid', how='left')
master_df = master_df.join(death_subset, on='personalid', how='left')

# Fill NaN for counts with 0
count_cols = ['total_prescriptions', 'unique_drugs', 'total_lab_tests', 'unique_test_types']
for col in count_cols:
    if col in master_df.columns:
        master_df = master_df.with_columns(pl.col(col).fill_null(0).cast(pl.Int64))

# Create additional features
print("Engineering additional features...")
master_df = master_df.with_columns([
    pl.when((pl.col('age') >= 0) & (pl.col('age') < 18))
    .then(pl.lit('<18'))
    .when((pl.col('age') >= 18) & (pl.col('age') < 30))
    .then(pl.lit('18-30'))
    .when((pl.col('age') >= 30) & (pl.col('age') < 40))
    .then(pl.lit('30-40'))
    .when((pl.col('age') >= 40) & (pl.col('age') <= 55))
    .then(pl.lit('40-55'))
    .when((pl.col('age') > 55) & (pl.col('age') <= 65))
    .then(pl.lit('55-65'))
    .otherwise(pl.lit('65+'))
    .alias('age_group'),
    
    ((pl.col('age') >= 40) & (pl.col('age') <= 55)).cast(pl.Int64).alias('in_target_age')
])

# Create comorbidity count
comorbidity_cols = [col for col in chronic_conditions if col in master_df.columns]
if comorbidity_cols:
    master_df = master_df.with_columns(
        pl.sum_horizontal([pl.col(col) for col in comorbidity_cols]).alias('comorbidity_count')
    )

# Create healthcare engagement score
visit_cols_map = {
    'visits_outpatient_total': 'total_outpatient_visits',
    'visits_inpatient_total': 'total_inpatient_visits',
    'visits_emergency_total': 'total_emergency_visits'
}

# Use correct column names
outpatient_col = 'total_outpatient_visits' if 'total_outpatient_visits' in master_df.columns else 'visits_outpatient_total'
inpatient_col = 'total_inpatient_visits' if 'total_inpatient_visits' in master_df.columns else 'visits_inpatient_total'
emergency_col = 'total_emergency_visits' if 'total_emergency_visits' in master_df.columns else 'visits_emergency_total'

if all(col in master_df.columns for col in [outpatient_col, inpatient_col, emergency_col]):
    master_df = master_df.with_columns(
        (pl.col(outpatient_col).fill_null(0) + 
         pl.col(inpatient_col).fill_null(0) * 3 +  # Weight inpatient more
         pl.col(emergency_col).fill_null(0) * 2).alias('healthcare_engagement')
    )
else:
    master_df = master_df.with_columns(pl.lit(0).alias('healthcare_engagement'))

print(f"\n[5.2] Combined Dataset Shape: {master_df.shape}")
print(f"Total individuals: {len(master_df):,}")
print(f"Total features: {len(master_df.columns)}")

# Save combined dataset
output_file = 'combined_dataset_polars.csv'
master_df.write_csv(output_file)
print(f"\n✓ Combined dataset saved to: {output_file}")

# Display sample
print("\n[5.3] Sample of combined dataset:")
print(master_df.head(10))

print("\n[5.4] Feature Summary:")
print(master_df.describe())

# Save detailed summary
summary_output = 'eda_outputs/data_summary_polars.txt'
with open(summary_output, 'w') as f:
    f.write("HEALTHCARE DATATHON - DATA SUMMARY (POLARS VERSION)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Generated: {datetime.now()}\n\n")
    
    f.write("DATASET SHAPES:\n")
    for name, df in datasets.items():
        f.write(f"  {name}: {df.shape}\n")
    
    f.write(f"\nCOMBINED DATASET: {master_df.shape}\n\n")
    
    f.write("DATA QUALITY ISSUES:\n")
    for issue in data_quality_report:
        f.write(f"  - {issue}\n")
    
    f.write("\n\nFEATURE DESCRIPTIONS:\n")
    f.write(str(master_df.describe()))

print(f"✓ Data summary saved to: {summary_output}")

# Final statistics for target population
print("\n" + "="*80)
print("TARGET POPULATION (Age 40-55) STATISTICS")
print("="*80)
target_pop = master_df.filter(pl.col('in_target_age') == 1)
print(f"\nTotal individuals in target age group: {len(target_pop):,}")
print(f"Percentage of total population: {len(target_pop)/len(master_df)*100:.2f}%")
print(f"\nChronic condition prevalence in target group:")
for condition in chronic_conditions:
    if condition in target_pop.columns:
        count = target_pop[condition].sum().item()
        pct = (count / len(target_pop)) * 100
        print(f"  {condition}: {count:,} ({pct:.2f}%)")

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nOutputs saved to:")
print(f"  - Combined dataset: {output_file}")
print(f"  - Plots: eda_outputs/plots/")
print(f"  - Summary: {summary_output}")

if USE_SAMPLING:
    print(f"\n⚠️  NOTE: This analysis used sampling ({SAMPLE_FRACTION*100:.1f}% of data)")
    print("   For full analysis, set USE_SAMPLING = False in the script")

print("\nMemory optimization features used:")
print(f"  - Polars for efficient data processing")
print(f"  - Chunk size: {CHUNK_SIZE:,} rows")
print(f"  - Garbage collection: Enabled")
print(f"  - Memory-efficient operations: Enabled")

print("\nReady for model development!")