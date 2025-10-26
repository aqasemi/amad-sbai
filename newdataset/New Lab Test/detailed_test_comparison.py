#!/usr/bin/env python3
"""
Detailed comparison of test types between new and original lab files
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")

# Define paths
NEW_LAB_DIR = Path("/home/ubuntu/datathon/newdataset/New Lab Test")
ORIG_LAB_DIR = Path("/home/ubuntu/datathon/dataset/LABs")

print("Loading test type information from samples...")

# Load samples from each file
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

# Load all test types
new_tests_dfs = []
for name, path in new_files.items():
    df = pd.read_csv(path, nrows=100000)
    df['source'] = f'NEW_{name}'
    new_tests_dfs.append(df[['test_code', 'test_name', 'source']])

orig_tests_dfs = []
for name, path in orig_files.items():
    df = pd.read_csv(path, nrows=100000)
    df['source'] = f'ORIG_{name}'
    orig_tests_dfs.append(df[['test_code', 'test_name', 'source']])

# Combine
new_all = pd.concat(new_tests_dfs).drop_duplicates(subset=['test_code', 'test_name'])
orig_all = pd.concat(orig_tests_dfs).drop_duplicates(subset=['test_code', 'test_name'])

# Get unique tests
new_tests = set(new_all['test_code'])
orig_tests = set(orig_all['test_code'])

common_tests = new_tests & orig_tests
new_only = new_tests - orig_tests
orig_only = orig_tests - new_tests

print(f"\nTest Types Summary:")
print(f"  Tests in both datasets: {len(common_tests)}")
print(f"  Tests only in NEW: {len(new_only)}")
print(f"  Tests only in ORIGINAL: {len(orig_only)}")

# Create visualization
fig = plt.figure(figsize=(18, 12))

# 1. Venn diagram-style visualization
ax1 = plt.subplot(2, 2, 1)
categories = ['Common\nTests', 'NEW\nOnly', 'ORIGINAL\nOnly']
counts = [len(common_tests), len(new_only), len(orig_only)]
colors = ['#95a5a6', '#2ecc71', '#3498db']
bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Test Types', fontsize=12)
ax1.set_title('Test Type Distribution', fontsize=14, fontweight='bold')
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# 2. Tests only in NEW
ax2 = plt.subplot(2, 2, 2)
ax2.axis('off')
new_only_info = new_all[new_all['test_code'].isin(new_only)][['test_code', 'test_name']].drop_duplicates()
text = "TESTS ONLY IN NEW FILES:\n" + "="*50 + "\n\n"
for idx, row in new_only_info.iterrows():
    text += f"• {row['test_code']}\n  {row['test_name']}\n\n"
ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
ax2.set_title('New Test Types', fontsize=14, fontweight='bold')

# 3. Tests only in ORIGINAL
ax3 = plt.subplot(2, 2, 3)
ax3.axis('off')
orig_only_info = orig_all[orig_all['test_code'].isin(orig_only)][['test_code', 'test_name']].drop_duplicates()
text = "TESTS ONLY IN ORIGINAL FILES:\n" + "="*50 + "\n\n"
for idx, row in orig_only_info.head(15).iterrows():
    name = row['test_name']
    if len(name) > 40:
        name = name[:37] + "..."
    text += f"• {row['test_code']}: {name}\n"
if len(orig_only_info) > 15:
    text += f"\n... and {len(orig_only_info) - 15} more tests"
ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
ax3.set_title('Missing Test Types (not in NEW)', fontsize=14, fontweight='bold')

# 4. Common tests pie chart
ax4 = plt.subplot(2, 2, 4)
pie_data = [len(common_tests), len(new_only), len(orig_only)]
pie_labels = [f'Common\n({len(common_tests)})', 
              f'NEW Only\n({len(new_only)})', 
              f'ORIG Only\n({len(orig_only)})']
colors_pie = ['#95a5a6', '#2ecc71', '#3498db']
wedges, texts, autotexts = ax4.pie(pie_data, labels=pie_labels, colors=colors_pie,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Test Type Overlap Proportion', fontsize=14, fontweight='bold')

plt.suptitle('Detailed Test Types Comparison: NEW vs ORIGINAL Lab Files', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save
output_file = NEW_LAB_DIR / "test_types_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved detailed comparison to: {output_file}")

# Create detailed CSV
comparison_data = []

# Common tests
for test_code in common_tests:
    test_name = new_all[new_all['test_code'] == test_code]['test_name'].iloc[0]
    comparison_data.append({
        'test_code': test_code,
        'test_name': test_name,
        'in_new': 'Yes',
        'in_original': 'Yes',
        'status': 'Common'
    })

# NEW only
for test_code in new_only:
    test_name = new_all[new_all['test_code'] == test_code]['test_name'].iloc[0]
    comparison_data.append({
        'test_code': test_code,
        'test_name': test_name,
        'in_new': 'Yes',
        'in_original': 'No',
        'status': 'NEW_ONLY'
    })

# ORIGINAL only
for test_code in orig_only:
    test_name = orig_all[orig_all['test_code'] == test_code]['test_name'].iloc[0]
    comparison_data.append({
        'test_code': test_code,
        'test_name': test_name,
        'in_new': 'No',
        'in_original': 'Yes',
        'status': 'ORIGINAL_ONLY'
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('status')
comparison_df.to_csv(NEW_LAB_DIR / "test_types_detailed_comparison.csv", index=False)
print(f"✓ Saved detailed test list to: {NEW_LAB_DIR / 'test_types_detailed_comparison.csv'}")

# Print detailed lists
print("\n" + "="*80)
print("DETAILED TEST LISTS")
print("="*80)

print(f"\n✅ Tests in BOTH datasets ({len(common_tests)}):")
common_info = new_all[new_all['test_code'].isin(common_tests)][['test_code', 'test_name']].drop_duplicates().sort_values('test_code')
for idx, row in common_info.iterrows():
    print(f"   {row['test_code']:15s} - {row['test_name']}")

print(f"\n🆕 Tests ONLY in NEW ({len(new_only)}):")
for idx, row in new_only_info.iterrows():
    print(f"   {row['test_code']:15s} - {row['test_name']}")

print(f"\n📊 Tests ONLY in ORIGINAL ({len(orig_only)}):")
for idx, row in orig_only_info.iterrows():
    print(f"   {row['test_code']:15s} - {row['test_name']}")

plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
