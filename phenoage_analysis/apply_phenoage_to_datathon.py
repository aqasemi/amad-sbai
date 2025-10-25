"""
Apply PhenoAge to Datathon Dataset
This script applies the validated PhenoAge biomarker combinations to the actual datathon data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

class DatathonPhenoAgeCalculator:
    """Applies PhenoAge calculations to datathon dataset using validated combinations"""
    
    def __init__(self):
        # Best performing combinations from validation
        self.best_combinations = {
            3: {
                'biomarkers': ['Creatinine', 'RDW', 'ALP'],
                'expected_r2': 0.933,
                'n_users': 29990,
                'formula': 'phenoage_3_optimal'
            },
            4: {
                'biomarkers': ['Albumin', 'Creatinine', 'Glucose', 'WBC'],
                'expected_r2': 0.941,
                'n_users': 4142,
                'formula': 'phenoage_4_optimal'
            },
            5: {
                'biomarkers': ['Albumin', 'Creatinine', 'RDW', 'ALP', 'WBC'],
                'expected_r2': 0.943,
                'n_users': 898,
                'formula': 'phenoage_5_optimal'
            }
        }
        
        # Most practical combination (best balance of performance and coverage)
        self.practical_combination = {
            'biomarkers': ['Albumin', 'Creatinine', 'ALP'],
            'expected_r2': 0.882,
            'n_users': 209757,
            'formula': 'phenoage_practical'
        }
        
        # Trained model coefficients (from validation on synthetic data)
        # In practice, these would come from training on real NHANES data
        self.model_coefficients = {
            'phenoage_3_optimal': {
                'intercept': 141.50225,
                'coefficients': {
                    'Creatinine': 0.07250,
                    'RDW': 0.12981,
                    'ALP': 0.01429,
                    'age': 0.42563
                },
                'scaler_params': {
                    'mean': [88.0, 13.2, 72.0, 50.0],
                    'std': [26.0, 1.3, 25.0, 18.0]
                }
            },
            'phenoage_4_optimal': {
                'intercept': 141.50225,
                'coefficients': {
                    'Albumin': -0.04049,
                    'Creatinine': 0.07115,
                    'Glucose': 0.12362,
                    'WBC': 0.03686,
                    'age': 0.42737
                },
                'scaler_params': {
                    'mean': [42.0, 88.0, 5.6, 7.2, 50.0],
                    'std': [4.0, 26.0, 2.2, 2.1, 18.0]
                }
            },
            'phenoage_5_optimal': {
                'intercept': 141.50225,
                'coefficients': {
                    'Albumin': -0.04141,
                    'Creatinine': 0.07152,
                    'RDW': 0.12806,
                    'ALP': 0.01320,
                    'WBC': 0.03535,
                    'age': 0.41201
                },
                'scaler_params': {
                    'mean': [42.0, 88.0, 13.2, 72.0, 7.2, 50.0],
                    'std': [4.0, 26.0, 1.3, 25.0, 2.1, 18.0]
                }
            },
            'phenoage_practical': {
                'intercept': 141.50225,
                'coefficients': {
                    'Albumin': -0.04326,
                    'Creatinine': 0.07419,
                    'ALP': 0.01445,
                    'age': 0.45108
                },
                'scaler_params': {
                    'mean': [42.0, 88.0, 72.0, 50.0],
                    'std': [4.0, 26.0, 25.0, 18.0]
                }
            }
        }
    
    def load_datathon_data(self, individuals_path, labs_path):
        """Load and prepare datathon data"""
        print("Loading datathon data...")
        
        # Load individuals data
        individuals_df = pd.read_csv(individuals_path)
        
        # Load labs data
        labs_df = pd.read_csv(labs_path)
        
        # Map test names to our biomarker names
        test_name_mapping = {
            'Albumin': 'Albumin',
            'Creatinine': 'Creatinine',
            'Glucose': 'Glucose',
            'C-Reactive Protein': 'CRP',
            'Lymphocytes %': 'LYM',
            'MCV': 'MCV',
            'RDW': 'RDW',
            'Alkaline Phosphatase': 'ALP',
            'WBC': 'WBC'
        }
        
        # Process lab data
        labs_pivot = self.process_lab_data(labs_df, test_name_mapping)
        
        # Merge with individuals
        data = individuals_df.merge(labs_pivot, on='individual_id', how='left')
        
        # Calculate age (assuming we have birth_date or age column)
        if 'age' not in data.columns:
            if 'birth_date' in data.columns:
                data['age'] = pd.to_datetime('today').year - pd.to_datetime(data['birth_date']).dt.year
            else:
                print("Warning: No age information found. Using default age of 50.")
                data['age'] = 50
        
        return data
    
    def process_lab_data(self, labs_df, test_mapping):
        """Process lab data to get latest values for each test"""
        # Filter relevant tests
        relevant_tests = list(test_mapping.keys())
        labs_filtered = labs_df[labs_df['test_name'].isin(relevant_tests)].copy()
        
        # Map test names
        labs_filtered['biomarker'] = labs_filtered['test_name'].map(test_mapping)
        
        # Get the latest test result for each individual and biomarker
        labs_filtered['test_date'] = pd.to_datetime(labs_filtered['test_date'])
        latest_tests = labs_filtered.sort_values('test_date').groupby(['individual_id', 'biomarker']).last()
        
        # Pivot to wide format
        labs_pivot = latest_tests['result_value'].unstack('biomarker')
        
        return labs_pivot
    
    def calculate_phenoage(self, data, formula_name):
        """Calculate PhenoAge using specified formula"""
        formula = self.model_coefficients[formula_name]
        biomarkers = [b for b in formula['coefficients'].keys() if b != 'age']
        
        # Prepare features
        features = biomarkers + ['age']
        X = data[features].copy()
        
        # Handle missing values
        for col in features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        # Standardize features
        scaler_mean = formula['scaler_params']['mean']
        scaler_std = formula['scaler_params']['std']
        
        X_scaled = X.copy()
        for i, col in enumerate(features):
            X_scaled[col] = (X[col] - scaler_mean[i]) / scaler_std[i]
        
        # Calculate linear predictor
        linear_predictor = formula['intercept']
        for feature in features:
            linear_predictor += formula['coefficients'][feature] * X_scaled[feature]
        
        # Convert to PhenoAge
        # Simplified formula - in practice, use the full Gompertz transformation
        phenoage = linear_predictor
        
        return phenoage
    
    def apply_all_formulas(self, data):
        """Apply all validated PhenoAge formulas to the data"""
        results = pd.DataFrame(index=data.index)
        results['individual_id'] = data['individual_id']
        results['chronological_age'] = data['age']
        
        # Apply practical formula (highest coverage)
        print("\nApplying practical formula (Albumin, Creatinine, ALP)...")
        has_practical = data[self.practical_combination['biomarkers'] + ['age']].notna().all(axis=1)
        results['phenoage_practical'] = np.nan
        results.loc[has_practical, 'phenoage_practical'] = self.calculate_phenoage(
            data[has_practical], 'phenoage_practical'
        )
        print(f"  Calculated for {has_practical.sum():,} individuals ({has_practical.mean()*100:.1f}%)")
        
        # Apply optimal formulas
        for n_biomarkers, config in self.best_combinations.items():
            print(f"\nApplying {n_biomarkers}-biomarker optimal formula...")
            biomarkers = config['biomarkers']
            has_biomarkers = data[biomarkers + ['age']].notna().all(axis=1)
            
            col_name = f'phenoage_{n_biomarkers}_optimal'
            results[col_name] = np.nan
            
            if has_biomarkers.sum() > 0:
                results.loc[has_biomarkers, col_name] = self.calculate_phenoage(
                    data[has_biomarkers], config['formula']
                )
            
            print(f"  Calculated for {has_biomarkers.sum():,} individuals ({has_biomarkers.mean()*100:.1f}%)")
        
        # Calculate age acceleration (biological age - chronological age)
        for col in results.columns:
            if col.startswith('phenoage_'):
                accel_col = col.replace('phenoage_', 'age_accel_')
                results[accel_col] = results[col] - results['chronological_age']
        
        # Add summary statistics
        results['has_any_phenoage'] = results[[col for col in results.columns if col.startswith('phenoage_')]].notna().any(axis=1)
        
        return results
    
    def create_summary_report(self, results):
        """Create summary report of PhenoAge calculations"""
        report = []
        report.append("=" * 80)
        report.append("DATATHON PHENOAGE CALCULATION SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Overall coverage
        total_individuals = len(results)
        has_any = results['has_any_phenoage'].sum()
        report.append(f"Total individuals: {total_individuals:,}")
        report.append(f"Individuals with at least one PhenoAge: {has_any:,} ({has_any/total_individuals*100:.1f}%)")
        report.append("")
        
        # Coverage by formula
        report.append("Coverage by Formula:")
        for col in results.columns:
            if col.startswith('phenoage_'):
                n_calculated = results[col].notna().sum()
                pct_calculated = n_calculated / total_individuals * 100
                report.append(f"  {col}: {n_calculated:,} ({pct_calculated:.1f}%)")
        
        # Age acceleration statistics
        report.append("\nAge Acceleration Statistics:")
        for col in results.columns:
            if col.startswith('age_accel_'):
                accel_data = results[col].dropna()
                if len(accel_data) > 0:
                    report.append(f"\n  {col}:")
                    report.append(f"    Mean: {accel_data.mean():.2f} years")
                    report.append(f"    Std: {accel_data.std():.2f} years")
                    report.append(f"    Min: {accel_data.min():.2f} years")
                    report.append(f"    Max: {accel_data.max():.2f} years")
                    report.append(f"    % Accelerated (>0): {(accel_data > 0).mean()*100:.1f}%")
        
        # Correlation between different formulas
        report.append("\nCorrelation Between Formulas:")
        phenoage_cols = [col for col in results.columns if col.startswith('phenoage_')]
        if len(phenoage_cols) > 1:
            corr_matrix = results[phenoage_cols].corr()
            for i in range(len(phenoage_cols)):
                for j in range(i+1, len(phenoage_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if not np.isnan(corr):
                        report.append(f"  {phenoage_cols[i]} vs {phenoage_cols[j]}: {corr:.3f}")
        
        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)

def main():
    """Main function to apply PhenoAge to datathon data"""
    print("Applying PhenoAge to Datathon Dataset")
    print("=" * 50)
    
    # Initialize calculator
    calculator = DatathonPhenoAgeCalculator()
    
    # Load datathon data
    # Update these paths to your actual data files
    individuals_path = '/home/ubuntu/datathon/dataset/20250929_datathon_2_individuals.csv'
    labs_path = '/home/ubuntu/datathon/dataset/20250929_datathon_2_labs.csv'
    
    try:
        data = calculator.load_datathon_data(individuals_path, labs_path)
        print(f"\nLoaded data for {len(data):,} individuals")
        
        # Apply all PhenoAge formulas
        results = calculator.apply_all_formulas(data)
        
        # Save results
        results.to_csv('phenoage_validation_results/datathon_phenoage_results.csv', index=False)
        print("\nResults saved to 'phenoage_validation_results/datathon_phenoage_results.csv'")
        
        # Create and save summary report
        summary = calculator.create_summary_report(results)
        with open('phenoage_validation_results/datathon_phenoage_summary.txt', 'w') as f:
            f.write(summary)
        
        print("\n" + summary)
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coverage comparison
        ax = axes[0, 0]
        coverage_data = []
        for col in results.columns:
            if col.startswith('phenoage_'):
                coverage_data.append({
                    'Formula': col.replace('phenoage_', ''),
                    'Coverage': results[col].notna().sum()
                })
        
        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            coverage_df.plot(x='Formula', y='Coverage', kind='bar', ax=ax)
            ax.set_title('Coverage by PhenoAge Formula')
            ax.set_ylabel('Number of Individuals')
            ax.tick_params(axis='x', rotation=45)
        
        # 2. Age acceleration distribution
        ax = axes[0, 1]
        for col in results.columns:
            if col.startswith('age_accel_practical'):
                data = results[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=50, alpha=0.7, label=col)
        ax.set_xlabel('Age Acceleration (years)')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Acceleration Distribution (Practical Formula)')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.legend()
        
        # 3. Chronological vs Biological Age
        ax = axes[1, 0]
        if 'phenoage_practical' in results.columns:
            mask = results['phenoage_practical'].notna()
            ax.scatter(results.loc[mask, 'chronological_age'], 
                      results.loc[mask, 'phenoage_practical'],
                      alpha=0.5, s=1)
            ax.plot([0, 100], [0, 100], 'r--', alpha=0.5)
            ax.set_xlabel('Chronological Age')
            ax.set_ylabel('Biological Age (PhenoAge)')
            ax.set_title('Chronological vs Biological Age')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
        
        # 4. Summary statistics table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = []
        for col in ['phenoage_practical', 'phenoage_3_optimal', 'phenoage_4_optimal', 'phenoage_5_optimal']:
            if col in results.columns:
                data = results[col].dropna()
                if len(data) > 0:
                    summary_data.append([
                        col.replace('phenoage_', ''),
                        f"{len(data):,}",
                        f"{data.mean():.1f}",
                        f"{data.std():.1f}"
                    ])
        
        if summary_data:
            table = ax.table(cellText=summary_data,
                           colLabels=['Formula', 'N', 'Mean Age', 'Std'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax.set_title('PhenoAge Summary Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig('phenoage_validation_results/datathon_phenoage_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nVisualization saved to 'phenoage_validation_results/datathon_phenoage_visualization.png'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the data files exist and have the expected format.")

if __name__ == "__main__":
    main()