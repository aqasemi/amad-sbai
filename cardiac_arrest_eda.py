#!/usr/bin/env python3
"""
Comprehensive EDA for Cardiac Arrest Deaths
============================================
This script performs in-depth analysis of individuals who died from cardiac arrest,
exploring risk factors, lab biomarkers, medications, and physical activity patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

# Define data directories
BASE_DIR = Path('/home/ubuntu/datathon')
DATA_DIR = BASE_DIR / 'data2' / 'healththon - data'
DEATHS_DIR = DATA_DIR / 'Deaths'
INDIVIDUALS_DIR = DATA_DIR / 'Individuals'
LABS_DIR = DATA_DIR / 'LABs'
STEPS_DIR = DATA_DIR / 'Steps'
MEDS_DIR = DATA_DIR / 'Medications'
OUTPUT_DIR = BASE_DIR / 'cardiac_arrest_analysis'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("CARDIAC ARREST DEATHS - COMPREHENSIVE EDA")
print("="*80)

# ============================================================================
# STEP 1: LOAD DEATH DATA AND IDENTIFY CARDIAC ARREST CASES
# ============================================================================
print("\n" + "="*80)
print("STEP 1: IDENTIFYING CARDIAC ARREST CASES")
print("="*80)

# Load death data
death_file = DEATHS_DIR / '20251002_Death Data Hashed.csv'
print(f"\nLoading death data from {death_file.name}...")
deaths_df = pd.read_csv(death_file, low_memory=False)
print(f"✓ Total death records: {len(deaths_df):,}")

# Identify cardiac arrest cases
# Search for cardiac arrest in both direct and underlying causes
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

# convert bmi from string to numeric
deaths_df['bmi'] = pd.to_numeric(deaths_df['bmi'], errors='coerce')

cardiac_deaths = deaths_df[deaths_df['is_cardiac_arrest'] == True].copy()
non_cardiac_deaths = deaths_df[deaths_df['is_cardiac_arrest'] == False].copy()

print(f"\n✓ Cardiac arrest deaths: {len(cardiac_deaths):,} ({len(cardiac_deaths)/len(deaths_df)*100:.2f}%)")
print(f"✓ Non-cardiac deaths: {len(non_cardiac_deaths):,} ({len(non_cardiac_deaths)/len(deaths_df)*100:.2f}%)")

# Get list of cardiac arrest patient IDs
cardiac_patient_ids = set(cardiac_deaths['personalid'].values)
print(f"✓ Unique cardiac arrest patients: {len(cardiac_patient_ids):,}")

# Print top causes
print("\nTop direct death causes in cardiac arrest group:")
direct_causes = cardiac_deaths['directdeathcasueicd10'].value_counts().head(10)
for i, (cause, count) in enumerate(direct_causes.items(), 1):
    pct = (count / len(cardiac_deaths)) * 100
    print(f"  {i:2d}. {cause}: {count:,} ({pct:.1f}%)")

print("\nTop underlying death causes in cardiac arrest group:")
underlying_causes = cardiac_deaths['underlyingdeathcauseicd10'].value_counts().head(10)
for i, (cause, count) in enumerate(underlying_causes.items(), 1):
    pct = (count / len(cardiac_deaths)) * 100
    print(f"  {i:2d}. {cause}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 2: DEMOGRAPHIC AND HEALTH CONDITION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DEMOGRAPHIC AND HEALTH CONDITION ANALYSIS")
print("="*80)

# Compare demographics between cardiac and non-cardiac deaths
print("\n" + "-"*60)
print("AGE COMPARISON")
print("-"*60)

cardiac_ages = cardiac_deaths['age'].dropna()
non_cardiac_ages = non_cardiac_deaths['age'].dropna()

print(f"\nCardiac Arrest Deaths:")
print(f"  Mean age: {cardiac_ages.mean():.2f} years")
print(f"  Median age: {cardiac_ages.median():.2f} years")
print(f"  Std: {cardiac_ages.std():.2f}")
print(f"  Range: {cardiac_ages.min():.0f} - {cardiac_ages.max():.0f}")

print(f"\nNon-Cardiac Deaths:")
print(f"  Mean age: {non_cardiac_ages.mean():.2f} years")
print(f"  Median age: {non_cardiac_ages.median():.2f} years")
print(f"  Std: {non_cardiac_ages.std():.2f}")
print(f"  Range: {non_cardiac_ages.min():.0f} - {non_cardiac_ages.max():.0f}")

# Gender distribution
print("\n" + "-"*60)
print("GENDER DISTRIBUTION")
print("-"*60)

cardiac_gender = cardiac_deaths['gender'].value_counts()
non_cardiac_gender = non_cardiac_deaths['gender'].value_counts()

print("\nCardiac Arrest Deaths:")
for gender, count in cardiac_gender.items():
    pct = (count / len(cardiac_deaths)) * 100
    print(f"  {gender}: {count:,} ({pct:.1f}%)")

print("\nNon-Cardiac Deaths:")
for gender, count in non_cardiac_gender.items():
    pct = (count / len(non_cardiac_deaths)) * 100
    print(f"  {gender}: {count:,} ({pct:.1f}%)")

# Health conditions comparison
print("\n" + "-"*60)
print("HEALTH CONDITIONS PREVALENCE")
print("-"*60)

health_conditions = ['has_dyslipidemia', 'has_hypertension', 'has_obesity', 'has_diabetes']

for condition in health_conditions:
    cardiac_condition = cardiac_deaths[cardiac_deaths[condition] == 1]
    non_cardiac_condition = non_cardiac_deaths[non_cardiac_deaths[condition] == 1]
    
    cardiac_pct = (len(cardiac_condition) / len(cardiac_deaths)) * 100
    non_cardiac_pct = (len(non_cardiac_condition) / len(non_cardiac_deaths)) * 100
    
    print(f"\n{condition.replace('has_', '').replace('_', ' ').title()}:")
    print(f"  Cardiac arrest: {len(cardiac_condition):,} ({cardiac_pct:.2f}%)")
    print(f"  Non-cardiac: {len(non_cardiac_condition):,} ({non_cardiac_pct:.2f}%)")
    print(f"  Difference: {cardiac_pct - non_cardiac_pct:+.2f}%")

# BMI comparison
print("\n" + "-"*60)
print("BMI COMPARISON")
print("-"*60)

cardiac_bmi = cardiac_deaths['bmi'].dropna()
non_cardiac_bmi = non_cardiac_deaths['bmi'].dropna()

print(f"\nCardiac Arrest Deaths (N={len(cardiac_bmi):,}):")
print(f"  Mean BMI: {cardiac_bmi.mean():.2f}")
print(f"  Median BMI: {cardiac_bmi.median():.2f}")
print(f"  Std: {cardiac_bmi.std():.2f}")

print(f"\nNon-Cardiac Deaths (N={len(non_cardiac_bmi):,}):")
print(f"  Mean BMI: {non_cardiac_bmi.mean():.2f}")
print(f"  Median BMI: {non_cardiac_bmi.median():.2f}")
print(f"  Std: {non_cardiac_bmi.std():.2f}")

# ============================================================================
# STEP 3: LOAD AND ANALYZE LAB DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ANALYZING LAB BIOMARKERS")
print("="*80)

# Define key biomarkers to analyze
key_biomarkers = {
    'Albumin': ['Albumin'],
    'Creatinine': ['Creatinine'],
    'Glucose': ['Glucose'],
    'LYM': ['Lymphocyte'],
    'MCV': ['MCV'],
    'RDW': ['RDW', 'Red Cell Distribution Width', 'Erythrocyte distribution width'],
    'ALP': ['Alkaline phosphatase', 'ALP'],
    'WBC': ['WBC', 'Leukocytes'],
    'Hemoglobin': ['Hemoglobin', 'Hb'],
    'Cholesterol': ['Cholesterol'],
    'Triglycerides': ['Triglyceride'],
    'HDL': ['HDL'],
    'LDL': ['LDL'],
    'Potassium': ['Potassium'],
    'Sodium': ['Sodium'],
    'Troponin': ['Troponin'],
    'BNP': ['BNP', 'Brain Natriuretic Peptide'],
    'CRP': ['C-Reactive Protein', 'CRP']
}

# Function to check if a test name matches keywords
def matches_biomarker(sub_test_name, keywords):
    if pd.isna(sub_test_name):
        return False
    sub_test_name_lower = str(sub_test_name).lower()
    return any(keyword.lower() in sub_test_name_lower for keyword in keywords)

# Load lab files
lab_files = [
    '20250929_datathon_2_labs_above_50_alive.csv',
    '20251019_datathon_2_labs_0_35_all.csv',
    '20251019_datathon_2_labs_36_50_all.csv',
    '20251019_datathon_2_labs_above_50_death.csv'
]

print("\nLoading lab data for cardiac arrest patients...")
cardiac_labs = []
biomarker_stats = {name: {'values': [], 'patient_count': 0} for name in key_biomarkers.keys()}

for lab_file in lab_files:
    file_path = LABS_DIR / lab_file
    if not file_path.exists():
        print(f"  ⚠ Warning: {lab_file} not found")
        continue
    
    print(f"  Processing {lab_file}...")
    
    # Read in chunks to save memory
    chunk_count = 0
    for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
        # Filter for cardiac arrest patients
        cardiac_chunk = chunk[chunk['personalid'].isin(cardiac_patient_ids)]
        
        if len(cardiac_chunk) > 0:
            cardiac_labs.append(cardiac_chunk)
            
            # Collect biomarker values
            for biomarker_name, keywords in key_biomarkers.items():
                mask = cardiac_chunk['sub_test_name'].apply(lambda x: matches_biomarker(x, keywords))
                matching_rows = cardiac_chunk[mask]
                
                if len(matching_rows) > 0:
                    # Extract numeric values
                    values = pd.to_numeric(matching_rows['result_value'], errors='coerce').dropna()
                    biomarker_stats[biomarker_name]['values'].extend(values.tolist())
                    biomarker_stats[biomarker_name]['patient_count'] += matching_rows['personalid'].nunique()
        
        chunk_count += 1

if cardiac_labs:
    cardiac_labs_df = pd.concat(cardiac_labs, ignore_index=True)
    print(f"\n✓ Total lab records for cardiac arrest patients: {len(cardiac_labs_df):,}")
    print(f"✓ Unique patients with lab data: {cardiac_labs_df['personalid'].nunique():,}")
    
    # Analyze biomarker statistics
    print("\n" + "-"*60)
    print("KEY BIOMARKER STATISTICS")
    print("-"*60)
    
    for biomarker_name, stats in biomarker_stats.items():
        if len(stats['values']) > 0:
            values = np.array(stats['values'])
            print(f"\n{biomarker_name}:")
            print(f"  Patients: {stats['patient_count']:,}")
            print(f"  Total measurements: {len(values):,}")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Median: {np.median(values):.2f}")
            print(f"  Std: {np.std(values):.2f}")
            print(f"  Range: {np.min(values):.2f} - {np.max(values):.2f}")
            print(f"  25th percentile: {np.percentile(values, 25):.2f}")
            print(f"  75th percentile: {np.percentile(values, 75):.2f}")
    
    # Save lab data
    cardiac_labs_file = OUTPUT_DIR / 'cardiac_arrest_lab_data.csv'
    cardiac_labs_df.to_csv(cardiac_labs_file, index=False)
    print(f"\n✓ Cardiac arrest lab data saved to: {cardiac_labs_file}")
else:
    print("\n⚠ No lab data found for cardiac arrest patients")
    cardiac_labs_df = pd.DataFrame()

# ============================================================================
# STEP 4: LOAD AND ANALYZE STEPS DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 4: ANALYZING PHYSICAL ACTIVITY (STEPS)")
print("="*80)

steps_files = [
    '20250930_datathon_2_steps.csv',
    '20250930_datathon_2_steps_death.csv'
]

print("\nLoading steps data for cardiac arrest patients...")
cardiac_steps = []

for steps_file in steps_files:
    file_path = STEPS_DIR / steps_file
    if not file_path.exists():
        print(f"  ⚠ Warning: {steps_file} not found")
        continue
    
    print(f"  Processing {steps_file}...")
    
    # Read in chunks
    for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
        cardiac_chunk = chunk[chunk['personalid'].isin(cardiac_patient_ids)]
        if len(cardiac_chunk) > 0:
            cardiac_steps.append(cardiac_chunk)

if cardiac_steps:
    cardiac_steps_df = pd.concat(cardiac_steps, ignore_index=True)
    print(f"\n✓ Total step records for cardiac arrest patients: {len(cardiac_steps_df):,}")
    print(f"✓ Unique patients with step data: {cardiac_steps_df['personalid'].nunique():,}")
    
    # Analyze step patterns
    if 'step_count' in cardiac_steps_df.columns:
        step_counts = pd.to_numeric(cardiac_steps_df['step_count'], errors='coerce').dropna()
        
        print("\n" + "-"*60)
        print("STEP COUNT STATISTICS")
        print("-"*60)
        print(f"  Total step records: {len(step_counts):,}")
        print(f"  Mean steps per day: {step_counts.mean():.2f}")
        print(f"  Median steps per day: {step_counts.median():.2f}")
        print(f"  Std: {step_counts.std():.2f}")
        print(f"  Range: {step_counts.min():.2f} - {step_counts.max():.2f}")
        print(f"  25th percentile: {step_counts.quantile(0.25):.2f}")
        print(f"  75th percentile: {step_counts.quantile(0.75):.2f}")
        
        # Calculate average steps per patient
        avg_steps_per_patient = cardiac_steps_df.groupby('personalid')['step_count'].mean()
        print(f"\n  Average steps per patient: {avg_steps_per_patient.mean():.2f}")
        print(f"  Median avg steps per patient: {avg_steps_per_patient.median():.2f}")
    
    # Save steps data
    cardiac_steps_file = OUTPUT_DIR / 'cardiac_arrest_steps_data.csv'
    cardiac_steps_df.to_csv(cardiac_steps_file, index=False)
    print(f"\n✓ Cardiac arrest steps data saved to: {cardiac_steps_file}")
else:
    print("\n⚠ No steps data found for cardiac arrest patients")
    cardiac_steps_df = pd.DataFrame()

# ============================================================================
# STEP 5: LOAD AND ANALYZE MEDICATIONS DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 5: ANALYZING MEDICATIONS")
print("="*80)

meds_files = [
    '20251019_datathon_2_Medications_0_35_all.csv',
    '20251019_datathon_2_Medications_36_50_all.csv',
    '20251019_datathon_2_Medications_above_50_alive.csv',
    '20251019_datathon_2_Medications_above_50_death.csv'
]

print("\nLoading medication data for cardiac arrest patients...")
cardiac_meds = []

for meds_file in meds_files:
    file_path = MEDS_DIR / meds_file
    if not file_path.exists():
        print(f"  ⚠ Warning: {meds_file} not found")
        continue
    
    print(f"  Processing {meds_file}...")
    
    # Read in chunks
    for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
        cardiac_chunk = chunk[chunk['personalid'].isin(cardiac_patient_ids)]
        if len(cardiac_chunk) > 0:
            cardiac_meds.append(cardiac_chunk)

if cardiac_meds:
    cardiac_meds_df = pd.concat(cardiac_meds, ignore_index=True)
    print(f"\n✓ Total medication records for cardiac arrest patients: {len(cardiac_meds_df):,}")
    print(f"✓ Unique patients with medication data: {cardiac_meds_df['personalid'].nunique():,}")
    
    # Analyze top medications
    print("\n" + "-"*60)
    print("TOP 20 MEDICATIONS")
    print("-"*60)
    
    if 'drug_name' in cardiac_meds_df.columns:
        top_meds = cardiac_meds_df['drug_name'].value_counts().head(20)
        for i, (med, count) in enumerate(top_meds.items(), 1):
            pct = (count / len(cardiac_meds_df)) * 100
            patients = cardiac_meds_df[cardiac_meds_df['drug_name'] == med]['personalid'].nunique()
            print(f"  {i:2d}. {med}: {count:,} prescriptions ({patients:,} patients, {pct:.1f}%)")
    
    # Save medications data
    cardiac_meds_file = OUTPUT_DIR / 'cardiac_arrest_medications_data.csv'
    cardiac_meds_df.to_csv(cardiac_meds_file, index=False)
    print(f"\n✓ Cardiac arrest medications data saved to: {cardiac_meds_file}")
else:
    print("\n⚠ No medications data found for cardiac arrest patients")
    cardiac_meds_df = pd.DataFrame()

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Demographics Comparison
print("\nCreating demographic comparisons...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cardiac Arrest vs Non-Cardiac Deaths - Demographic Comparison', 
             fontsize=16, fontweight='bold')

# Age distribution
ax1 = axes[0, 0]
ax1.hist([cardiac_ages, non_cardiac_ages], bins=30, label=['Cardiac Arrest', 'Non-Cardiac'], 
         alpha=0.7, color=['red', 'blue'], edgecolor='black')
ax1.set_xlabel('Age (years)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Age Distribution Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Gender distribution
ax2 = axes[0, 1]
gender_comparison = pd.DataFrame({
    'Cardiac Arrest': cardiac_deaths['gender'].value_counts(normalize=True) * 100,
    'Non-Cardiac': non_cardiac_deaths['gender'].value_counts(normalize=True) * 100
})
gender_comparison.plot(kind='bar', ax=ax2, color=['red', 'blue'], edgecolor='black', alpha=0.7)
ax2.set_xlabel('Gender', fontsize=11)
ax2.set_ylabel('Percentage (%)', fontsize=11)
ax2.set_title('Gender Distribution Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3, axis='y')

# Health conditions comparison
ax3 = axes[1, 0]
cardiac_conditions = []
non_cardiac_conditions = []
condition_labels = []

for condition in health_conditions:
    cardiac_pct = (cardiac_deaths[cardiac_deaths[condition] == 1].shape[0] / len(cardiac_deaths)) * 100
    non_cardiac_pct = (non_cardiac_deaths[non_cardiac_deaths[condition] == 1].shape[0] / len(non_cardiac_deaths)) * 100
    cardiac_conditions.append(cardiac_pct)
    non_cardiac_conditions.append(non_cardiac_pct)
    condition_labels.append(condition.replace('has_', '').replace('_', ' ').title())

x = np.arange(len(condition_labels))
width = 0.35

bars1 = ax3.bar(x - width/2, cardiac_conditions, width, label='Cardiac Arrest', 
                color='red', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x + width/2, non_cardiac_conditions, width, label='Non-Cardiac', 
                color='blue', alpha=0.7, edgecolor='black')

ax3.set_xlabel('Health Condition', fontsize=11)
ax3.set_ylabel('Prevalence (%)', fontsize=11)
ax3.set_title('Health Conditions Prevalence Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(condition_labels, rotation=45, ha='right')
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# BMI comparison
ax4 = axes[1, 1]
ax4.hist([cardiac_bmi, non_cardiac_bmi], bins=30, label=['Cardiac Arrest', 'Non-Cardiac'],
         alpha=0.7, color=['red', 'blue'], edgecolor='black')
ax4.axvline(cardiac_bmi.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Cardiac Mean: {cardiac_bmi.mean():.1f}')
ax4.axvline(non_cardiac_bmi.mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Non-Cardiac Mean: {non_cardiac_bmi.mean():.1f}')
ax4.set_xlabel('BMI', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('BMI Distribution Comparison', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
demographics_plot = OUTPUT_DIR / 'cardiac_arrest_demographics_comparison.png'
plt.savefig(demographics_plot, dpi=300, bbox_inches='tight')
print(f"✓ Demographics comparison saved to: {demographics_plot}")
plt.close()

# Visualization 2: Biomarker Analysis
if len(cardiac_labs_df) > 0:
    print("\nCreating biomarker distributions...")
    
    # Filter biomarkers with data
    biomarkers_with_data = {k: v for k, v in biomarker_stats.items() if len(v['values']) > 0}
    
    if len(biomarkers_with_data) > 0:
        n_biomarkers = min(12, len(biomarkers_with_data))
        n_rows = (n_biomarkers + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
        fig.suptitle('Key Biomarker Distributions in Cardiac Arrest Patients', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_biomarkers > 1 else [axes]
        
        for idx, (biomarker_name, stats) in enumerate(list(biomarkers_with_data.items())[:n_biomarkers]):
            ax = axes[idx]
            values = np.array(stats['values'])
            
            # Remove outliers for better visualization (using IQR method)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            ax.hist(filtered_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(values):.2f}')
            ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(values):.2f}')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'{biomarker_name}\n(N={stats["patient_count"]:,} patients)', 
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_biomarkers, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        biomarkers_plot = OUTPUT_DIR / 'cardiac_arrest_biomarkers_distribution.png'
        plt.savefig(biomarkers_plot, dpi=300, bbox_inches='tight')
        print(f"✓ Biomarker distributions saved to: {biomarkers_plot}")
        plt.close()

# Visualization 3: Steps Analysis
if len(cardiac_steps_df) > 0 and 'step_count' in cardiac_steps_df.columns:
    print("\nCreating steps analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Physical Activity Patterns in Cardiac Arrest Patients', 
                 fontsize=16, fontweight='bold')
    
    step_counts = pd.to_numeric(cardiac_steps_df['step_count'], errors='coerce').dropna()
    
    # Overall distribution
    ax1 = axes[0]
    # Filter extreme outliers for better visualization
    q1, q3 = step_counts.quantile([0.25, 0.75])
    iqr = q3 - q1
    filtered_steps = step_counts[(step_counts >= q1 - 3*iqr) & (step_counts <= q3 + 3*iqr)]
    
    ax1.hist(filtered_steps, bins=50, color='green', edgecolor='black', alpha=0.7)
    ax1.axvline(step_counts.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {step_counts.mean():.0f}')
    ax1.axvline(step_counts.median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {step_counts.median():.0f}')
    ax1.set_xlabel('Daily Steps', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Daily Step Count Distribution\n(N={len(step_counts):,} records)', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Average steps per patient
    ax2 = axes[1]
    avg_steps = cardiac_steps_df.groupby('personalid')['step_count'].mean()
    ax2.hist(avg_steps, bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax2.axvline(avg_steps.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {avg_steps.mean():.0f}')
    ax2.axvline(avg_steps.median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {avg_steps.median():.0f}')
    ax2.set_xlabel('Average Daily Steps', fontsize=11)
    ax2.set_ylabel('Number of Patients', fontsize=11)
    ax2.set_title(f'Average Steps Per Patient\n(N={len(avg_steps):,} patients)', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    steps_plot = OUTPUT_DIR / 'cardiac_arrest_steps_analysis.png'
    plt.savefig(steps_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Steps analysis saved to: {steps_plot}")
    plt.close()

# Visualization 4: Top Medications
if len(cardiac_meds_df) > 0 and 'drug_name' in cardiac_meds_df.columns:
    print("\nCreating medications analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Medication Patterns in Cardiac Arrest Patients', 
                 fontsize=16, fontweight='bold')
    
    # Top medications by prescription count
    ax1 = axes[0]
    top_meds = cardiac_meds_df['drug_name'].value_counts().head(20)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_meds)))
    top_meds.plot(kind='barh', ax=ax1, color=colors, edgecolor='black')
    ax1.set_xlabel('Number of Prescriptions', fontsize=11)
    ax1.set_ylabel('Medication', fontsize=11)
    ax1.set_title('Top 20 Medications by Prescription Count', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(alpha=0.3, axis='x')
    
    # Top medications by patient count
    ax2 = axes[1]
    meds_by_patient = cardiac_meds_df.groupby('drug_name')['personalid'].nunique().sort_values(ascending=False).head(20)
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(meds_by_patient)))
    meds_by_patient.plot(kind='barh', ax=ax2, color=colors2, edgecolor='black')
    ax2.set_xlabel('Number of Unique Patients', fontsize=11)
    ax2.set_ylabel('Medication', fontsize=11)
    ax2.set_title('Top 20 Medications by Patient Count', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    meds_plot = OUTPUT_DIR / 'cardiac_arrest_medications_analysis.png'
    plt.savefig(meds_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Medications analysis saved to: {meds_plot}")
    plt.close()

# ============================================================================
# STEP 7: GENERATE COMPREHENSIVE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 7: GENERATING SUMMARY REPORT")
print("="*80)

summary_report = OUTPUT_DIR / 'cardiac_arrest_analysis_summary.txt'

with open(summary_report, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CARDIAC ARREST DEATHS - COMPREHENSIVE ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. OVERVIEW\n")
    f.write("-"*60 + "\n")
    f.write(f"Total deaths analyzed: {len(deaths_df):,}\n")
    f.write(f"Cardiac arrest deaths: {len(cardiac_deaths):,} ({len(cardiac_deaths)/len(deaths_df)*100:.2f}%)\n")
    f.write(f"Non-cardiac deaths: {len(non_cardiac_deaths):,} ({len(non_cardiac_deaths)/len(deaths_df)*100:.2f}%)\n\n")
    
    f.write("2. DEMOGRAPHIC INSIGHTS\n")
    f.write("-"*60 + "\n")
    f.write(f"Age - Cardiac Arrest:\n")
    f.write(f"  Mean: {cardiac_ages.mean():.2f} years\n")
    f.write(f"  Median: {cardiac_ages.median():.2f} years\n")
    f.write(f"Age - Non-Cardiac:\n")
    f.write(f"  Mean: {non_cardiac_ages.mean():.2f} years\n")
    f.write(f"  Median: {non_cardiac_ages.median():.2f} years\n\n")
    
    f.write("Gender Distribution - Cardiac Arrest:\n")
    for gender, count in cardiac_gender.items():
        pct = (count / len(cardiac_deaths)) * 100
        f.write(f"  {gender}: {count:,} ({pct:.1f}%)\n")
    f.write("\n")
    
    f.write("3. RISK FACTORS (Health Conditions)\n")
    f.write("-"*60 + "\n")
    for condition in health_conditions:
        cardiac_pct = (cardiac_deaths[cardiac_deaths[condition] == 1].shape[0] / len(cardiac_deaths)) * 100
        non_cardiac_pct = (non_cardiac_deaths[non_cardiac_deaths[condition] == 1].shape[0] / len(non_cardiac_deaths)) * 100
        diff = cardiac_pct - non_cardiac_pct
        f.write(f"{condition.replace('has_', '').replace('_', ' ').title()}:\n")
        f.write(f"  Cardiac arrest: {cardiac_pct:.2f}%\n")
        f.write(f"  Non-cardiac: {non_cardiac_pct:.2f}%\n")
        f.write(f"  Difference: {diff:+.2f}%\n\n")
    
    if len(cardiac_bmi) > 0:
        f.write("BMI:\n")
        f.write(f"  Cardiac arrest - Mean: {cardiac_bmi.mean():.2f}\n")
        f.write(f"  Non-cardiac - Mean: {non_cardiac_bmi.mean():.2f}\n\n")
    
    f.write("4. LAB BIOMARKERS\n")
    f.write("-"*60 + "\n")
    if len(cardiac_labs_df) > 0:
        f.write(f"Total lab records: {len(cardiac_labs_df):,}\n")
        f.write(f"Patients with lab data: {cardiac_labs_df['personalid'].nunique():,}\n\n")
        
        f.write("Key Biomarker Statistics:\n")
        for biomarker_name, stats in biomarker_stats.items():
            if len(stats['values']) > 0:
                values = np.array(stats['values'])
                f.write(f"\n{biomarker_name}:\n")
                f.write(f"  Patients: {stats['patient_count']:,}\n")
                f.write(f"  Measurements: {len(values):,}\n")
                f.write(f"  Mean: {np.mean(values):.2f}\n")
                f.write(f"  Median: {np.median(values):.2f}\n")
                f.write(f"  Range: {np.min(values):.2f} - {np.max(values):.2f}\n")
    else:
        f.write("No lab data available for cardiac arrest patients\n\n")
    
    f.write("\n5. PHYSICAL ACTIVITY (STEPS)\n")
    f.write("-"*60 + "\n")
    if len(cardiac_steps_df) > 0:
        f.write(f"Total step records: {len(cardiac_steps_df):,}\n")
        f.write(f"Patients with step data: {cardiac_steps_df['personalid'].nunique():,}\n")
        if 'step_count' in cardiac_steps_df.columns:
            step_counts = pd.to_numeric(cardiac_steps_df['step_count'], errors='coerce').dropna()
            f.write(f"Mean steps per day: {step_counts.mean():.2f}\n")
            f.write(f"Median steps per day: {step_counts.median():.2f}\n")
    else:
        f.write("No steps data available for cardiac arrest patients\n")
    
    f.write("\n6. MEDICATIONS\n")
    f.write("-"*60 + "\n")
    if len(cardiac_meds_df) > 0:
        f.write(f"Total medication records: {len(cardiac_meds_df):,}\n")
        f.write(f"Patients with medication data: {cardiac_meds_df['personalid'].nunique():,}\n\n")
        
        if 'drug_name' in cardiac_meds_df.columns:
            f.write("Top 20 Medications:\n")
            top_meds = cardiac_meds_df['drug_name'].value_counts().head(20)
            for i, (med, count) in enumerate(top_meds.items(), 1):
                patients = cardiac_meds_df[cardiac_meds_df['drug_name'] == med]['personalid'].nunique()
                f.write(f"  {i:2d}. {med}: {count:,} prescriptions ({patients:,} patients)\n")
    else:
        f.write("No medications data available for cardiac arrest patients\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS AND RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write("POTENTIAL RISK INDICATORS FOR CARDIAC ARREST:\n\n")
    
    # Identify strongest risk factors
    f.write("1. Health Conditions:\n")
    condition_risks = []
    for condition in health_conditions:
        cardiac_pct = (cardiac_deaths[cardiac_deaths[condition] == 1].shape[0] / len(cardiac_deaths)) * 100
        non_cardiac_pct = (non_cardiac_deaths[non_cardiac_deaths[condition] == 1].shape[0] / len(non_cardiac_deaths)) * 100
        diff = cardiac_pct - non_cardiac_pct
        condition_risks.append((condition, diff, cardiac_pct))
    
    condition_risks.sort(key=lambda x: x[1], reverse=True)
    for condition, diff, prevalence in condition_risks:
        if diff > 0:
            f.write(f"   - {condition.replace('has_', '').replace('_', ' ').title()}: ")
            f.write(f"{diff:+.2f}% higher in cardiac arrest group (prevalence: {prevalence:.1f}%)\n")
    
    f.write("\n2. Biomarkers to Monitor:\n")
    if biomarker_stats:
        biomarkers_sorted = sorted(
            [(name, stats) for name, stats in biomarker_stats.items() if len(stats['values']) > 0],
            key=lambda x: x[1]['patient_count'], 
            reverse=True
        )
        for biomarker_name, stats in biomarkers_sorted[:10]:
            f.write(f"   - {biomarker_name}: Available for {stats['patient_count']:,} patients\n")
    
    f.write("\n3. Lifestyle Factors:\n")
    if len(cardiac_bmi) > 0:
        f.write(f"   - BMI: Mean {cardiac_bmi.mean():.2f}\n")
    if len(cardiac_steps_df) > 0 and 'step_count' in cardiac_steps_df.columns:
        step_counts = pd.to_numeric(cardiac_steps_df['step_count'], errors='coerce').dropna()
        f.write(f"   - Physical activity: Mean {step_counts.mean():.0f} steps/day\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"\n✓ Summary report saved to: {summary_report}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print(f"  1. cardiac_arrest_demographics_comparison.png")
print(f"  2. cardiac_arrest_biomarkers_distribution.png")
print(f"  3. cardiac_arrest_steps_analysis.png")
print(f"  4. cardiac_arrest_medications_analysis.png")
print(f"  5. cardiac_arrest_lab_data.csv")
print(f"  6. cardiac_arrest_steps_data.csv")
print(f"  7. cardiac_arrest_medications_data.csv")
print(f"  8. cardiac_arrest_analysis_summary.txt")
print("\n" + "="*80)
