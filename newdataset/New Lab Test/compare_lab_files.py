#!/usr/bin/env python3
"""
Compare new lab test files with original lab files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define paths
NEW_LAB_DIR = Path("/home/ubuntu/datathon/newdataset/New Lab Test")
ORIG_LAB_DIR = Path("/home/ubuntu/datathon/dataset/LABs")

print("=" * 80)
print("LAB FILES COMPARISON ANALYSIS")
print("=" * 80)

# Load file information
new_files = {
    "above_0_30": NEW_LAB_DIR / "20251023_datathon_2_labs_above_0_30_all.csv",
    "31_50": NEW_LAB_DIR / "20251023_datathon_2_labs_31_50_all.csv",
    "above_50": NEW_LAB_DIR / "20251023_datathon_2_labs_above_50_all.csv"
}

orig_files = {
    "0_35": ORIG_LAB_DIR / "20251019_datathon_2_labs_0_35_all.csv",
    "36_50": ORIG_LAB_DIR / "20251019_datathon_2_labs_36_50_all.csv",
    "above_50_alive": ORIG_LAB_DIR / "20250929_datathon_2_labs_above_50_alive.csv",
    "above_50_death": ORIG_LAB_DIR / "20251019_datathon_2_labs_above_50_death.csv"
}

# Function to get basic stats without loading full file
def get_file_stats(file_path, sample_size=100000):
    """Get basic statistics from a CSV file"""
    print(f"\nAnalyzing: {file_path.name}")
    
    # Get row count
    with open(file_path, 'r') as f:
        row_count = sum(1 for line in f) - 1  # Subtract header
    
    # Load sample
    df_sample = pd.read_csv(file_path, nrows=sample_size)
    
    stats = {
        'file': file_path.name,
        'total_rows': row_count,
        'columns': list(df_sample.columns),
        'unique_patients': df_sample['personalid'].nunique(),
        'unique_tests': df_sample['test_code'].nunique(),
        'unique_test_names': df_sample['test_name'].nunique(),
        'date_range': (df_sample['order_date'].min(), df_sample['order_date'].max()),
        'sample_size': len(df_sample)
    }
    
    print(f"  Total rows: {row_count:,}")
    print(f"  Unique patients (sample): {stats['unique_patients']:,}")
    print(f"  Unique test codes (sample): {stats['unique_tests']:,}")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    
    return stats, df_sample

# Analyze new files
print("\n" + "="*80)
print("NEW LAB FILES (October 23, 2023)")
print("="*80)

new_stats = {}
new_samples = {}

for key, file_path in new_files.items():
    stats, sample = get_file_stats(file_path)
    new_stats[key] = stats
    new_samples[key] = sample

# Analyze original files
print("\n" + "="*80)
print("ORIGINAL LAB FILES (October 19, 2023)")
print("="*80)

orig_stats = {}
orig_samples = {}

for key, file_path in orig_files.items():
    stats, sample = get_file_stats(file_path)
    orig_stats[key] = stats
    orig_samples[key] = sample

# Create comparison summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

# Total rows comparison
print("\n1. TOTAL RECORDS:")
new_total = sum(s['total_rows'] for s in new_stats.values())
orig_total = sum(s['total_rows'] for s in orig_stats.values())
print(f"   NEW files total:      {new_total:,} rows")
print(f"   ORIGINAL files total: {orig_total:,} rows")
print(f"   Difference:           {new_total - orig_total:,} rows ({(new_total/orig_total - 1)*100:.2f}%)")

# Segmentation comparison
print("\n2. FILE SEGMENTATION:")
print("\n   NEW files (by age ranges):")
for key, stats in new_stats.items():
    print(f"   - {key:20s}: {stats['total_rows']:>10,} rows")

print("\n   ORIGINAL files (by age ranges + death status):")
for key, stats in orig_stats.items():
    print(f"   - {key:20s}: {stats['total_rows']:>10,} rows")

# Test types comparison
print("\n3. TEST TYPES (from samples):")
all_new_tests = pd.concat([s[['test_code', 'test_name']] for s in new_samples.values()]).drop_duplicates()
all_orig_tests = pd.concat([s[['test_code', 'test_name']] for s in orig_samples.values()]).drop_duplicates()

print(f"   NEW files:      {len(all_new_tests)} unique test types")
print(f"   ORIGINAL files: {len(all_orig_tests)} unique test types")

# Find differences in test types
new_only_tests = set(all_new_tests['test_code']) - set(all_orig_tests['test_code'])
orig_only_tests = set(all_orig_tests['test_code']) - set(all_new_tests['test_code'])

if new_only_tests:
    print(f"\n   Tests only in NEW: {len(new_only_tests)}")
    print(f"   Examples: {list(new_only_tests)[:5]}")

if orig_only_tests:
    print(f"\n   Tests only in ORIGINAL: {len(orig_only_tests)}")
    print(f"   Examples: {list(orig_only_tests)[:5]}")

# Patient overlap analysis
print("\n4. PATIENT OVERLAP (from samples):")
all_new_patients = set()
for sample in new_samples.values():
    all_new_patients.update(sample['personalid'].unique())

all_orig_patients = set()
for sample in orig_samples.values():
    all_orig_patients.update(sample['personalid'].unique())

overlap_patients = all_new_patients & all_orig_patients
print(f"   Unique patients in NEW:      {len(all_new_patients):,}")
print(f"   Unique patients in ORIGINAL: {len(all_orig_patients):,}")
print(f"   Overlap:                     {len(overlap_patients):,} ({len(overlap_patients)/len(all_orig_patients)*100:.1f}%)")

# Create visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# 1. File sizes comparison
ax1 = plt.subplot(2, 3, 1)
files_data = []
for key, stats in new_stats.items():
    files_data.append({'File': f"NEW: {key}", 'Rows': stats['total_rows'], 'Type': 'New'})
for key, stats in orig_stats.items():
    files_data.append({'File': f"ORIG: {key}", 'Rows': stats['total_rows'], 'Type': 'Original'})

df_files = pd.DataFrame(files_data)
colors = {'New': '#2ecc71', 'Original': '#3498db'}
bars = ax1.barh(df_files['File'], df_files['Rows'], 
                color=[colors[t] for t in df_files['Type']])
ax1.set_xlabel('Number of Rows', fontsize=12)
ax1.set_title('File Sizes Comparison', fontsize=14, fontweight='bold')
ax1.ticklabel_format(style='plain', axis='x')
for i, (v, row) in enumerate(zip(df_files['Rows'], df_files.iterrows())):
    ax1.text(v, i, f' {v:,}', va='center', fontsize=9)

# 2. Total records by dataset
ax2 = plt.subplot(2, 3, 2)
totals = pd.DataFrame({
    'Dataset': ['New Files', 'Original Files'],
    'Total Rows': [new_total, orig_total]
})
bars = ax2.bar(totals['Dataset'], totals['Total Rows'], color=['#2ecc71', '#3498db'])
ax2.set_ylabel('Number of Rows', fontsize=12)
ax2.set_title('Total Records Comparison', fontsize=14, fontweight='bold')
ax2.ticklabel_format(style='plain', axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Top test types in new files
ax3 = plt.subplot(2, 3, 3)
combined_new = pd.concat(new_samples.values())
top_tests_new = combined_new['test_name'].value_counts().head(15)
ax3.barh(range(len(top_tests_new)), top_tests_new.values, color='#2ecc71')
ax3.set_yticks(range(len(top_tests_new)))
ax3.set_yticklabels([name[:40] + '...' if len(name) > 40 else name 
                      for name in top_tests_new.index], fontsize=8)
ax3.set_xlabel('Count (in sample)', fontsize=12)
ax3.set_title('Top 15 Lab Tests in NEW Files', fontsize=14, fontweight='bold')

# 4. Top test types in original files
ax4 = plt.subplot(2, 3, 4)
combined_orig = pd.concat(orig_samples.values())
top_tests_orig = combined_orig['test_name'].value_counts().head(15)
ax4.barh(range(len(top_tests_orig)), top_tests_orig.values, color='#3498db')
ax4.set_yticks(range(len(top_tests_orig)))
ax4.set_yticklabels([name[:40] + '...' if len(name) > 40 else name 
                      for name in top_tests_orig.index], fontsize=8)
ax4.set_xlabel('Count (in sample)', fontsize=12)
ax4.set_title('Top 15 Lab Tests in ORIGINAL Files', fontsize=14, fontweight='bold')

# 5. Segmentation visualization
ax5 = plt.subplot(2, 3, 5)
seg_data = []
seg_data.append({'Range': '0-30', 'Rows': new_stats['above_0_30']['total_rows'], 'Dataset': 'New'})
seg_data.append({'Range': '31-50', 'Rows': new_stats['31_50']['total_rows'], 'Dataset': 'New'})
seg_data.append({'Range': '>50', 'Rows': new_stats['above_50']['total_rows'], 'Dataset': 'New'})
seg_data.append({'Range': '0-35', 'Rows': orig_stats['0_35']['total_rows'], 'Dataset': 'Original'})
seg_data.append({'Range': '36-50', 'Rows': orig_stats['36_50']['total_rows'], 'Dataset': 'Original'})
seg_data.append({'Range': '>50 alive', 'Rows': orig_stats['above_50_alive']['total_rows'], 'Dataset': 'Original'})
seg_data.append({'Range': '>50 death', 'Rows': orig_stats['above_50_death']['total_rows'], 'Dataset': 'Original'})
df_seg = pd.DataFrame(seg_data)

x_new = [0, 1, 2]
x_orig = [4, 5, 6, 7]
ax5.bar(x_new, df_seg[df_seg['Dataset'] == 'New']['Rows'], 
        color='#2ecc71', label='New', width=0.8)
ax5.bar(x_orig, df_seg[df_seg['Dataset'] == 'Original']['Rows'], 
        color='#3498db', label='Original', width=0.8)
ax5.set_xticks(x_new + x_orig)
ax5.set_xticklabels(['0-30', '31-50', '>50', '0-35', '36-50', '>50\nalive', '>50\ndeath'], 
                     fontsize=9)
ax5.set_ylabel('Number of Rows', fontsize=12)
ax5.set_title('Age Range Segmentation', fontsize=14, fontweight='bold')
ax5.legend()
ax5.ticklabel_format(style='plain', axis='y')

# 6. Key differences summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
KEY DIFFERENCES SUMMARY

File Organization:
  NEW: 3 files by age (0-30, 31-50, >50)
  ORIGINAL: 4 files by age + death status

Total Records:
  NEW: {new_total:,} rows
  ORIGINAL: {orig_total:,} rows
  Difference: {abs(new_total - orig_total):,} rows

Segmentation Changes:
  • NEW splits 0-50 at age 30 (not 35)
  • NEW combines all >50 regardless of status
  • ORIGINAL separates >50 by alive/death

Patient Coverage (sample):
  • {len(overlap_patients):,} patients appear in both
  • {len(all_new_patients - all_orig_patients):,} only in NEW
  • {len(all_orig_patients - all_new_patients):,} only in ORIGINAL

Test Types:
  • NEW has {len(all_new_tests)} unique test types
  • ORIGINAL has {len(all_orig_tests)} unique test types
"""

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('New Lab Files vs Original Lab Files - Comprehensive Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_file = NEW_LAB_DIR / "lab_files_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_file}")

# Create detailed test comparison
print("\n" + "="*80)
print("DETAILED TEST COMPARISON")
print("="*80)

# Most common tests in each dataset
print("\nTop 10 most common tests in NEW files:")
for i, (test, count) in enumerate(top_tests_new.head(10).items(), 1):
    print(f"  {i:2d}. {test[:60]:60s} ({count:,})")

print("\nTop 10 most common tests in ORIGINAL files:")
for i, (test, count) in enumerate(top_tests_orig.head(10).items(), 1):
    print(f"  {i:2d}. {test[:60]:60s} ({count:,})")

# Save detailed comparison to CSV
comparison_df = pd.DataFrame({
    'Metric': ['Total Rows', 'Unique Tests (sample)', 'Unique Patients (sample)', 'Files'],
    'New Files': [new_total, len(all_new_tests), len(all_new_patients), len(new_files)],
    'Original Files': [orig_total, len(all_orig_tests), len(all_orig_patients), len(orig_files)]
})
comparison_df.to_csv(NEW_LAB_DIR / "comparison_summary.csv", index=False)
print(f"\n✓ Saved comparison summary to: {NEW_LAB_DIR / 'comparison_summary.csv'}")

plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
