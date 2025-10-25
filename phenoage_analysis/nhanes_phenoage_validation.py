"""
NHANES PhenoAge Validation for Datathon Biomarker Combinations
This script validates which combinations of commonly available biomarkers 
can provide reliable biological age estimation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
os.makedirs('phenoage_validation_results', exist_ok=True)

class DatathonPhenoAgeValidator:
    """Validates biomarker combinations for PhenoAge estimation"""
    
    def __init__(self):
        # PhenoAge coefficients from Levine et al. 2018
        self.phenoage_weights = {
            'albumin': -0.0336,
            'creatinine': 0.0095,
            'glucose': 0.1953,
            'crp_log': 0.0954,
            'lymphocyte_pct': -0.0120,
            'mcv': 0.0268,
            'rdw': 0.3306,
            'alkaline_phosphatase': 0.0019,
            'wbc': 0.0554,
            'age': 0.0804
        }
        
        # Biomarker mapping (simplified names to match datathon)
        self.biomarker_map = {
            'albumin': 'Albumin',
            'creatinine': 'Creatinine', 
            'glucose': 'Glucose',
            'crp': 'CRP',
            'lymphocyte_pct': 'LYM',
            'mcv': 'MCV',
            'rdw': 'RDW',
            'alkaline_phosphatase': 'ALP',
            'wbc': 'WBC'
        }
        
        # Datathon combinations with user counts
        self.datathon_combinations = {
            3: [
                (['Albumin', 'Creatinine', 'ALP'], 209757),
                (['Creatinine', 'ALP', 'WBC'], 112089),
                (['Albumin', 'Creatinine', 'WBC'], 83543),
                (['Albumin', 'ALP', 'WBC'], 75431),
                (['Creatinine', 'LYM', 'WBC'], 46978),
                (['Creatinine', 'MCV', 'ALP'], 36234),
                (['Creatinine', 'LYM', 'ALP'], 30564),
                (['Creatinine', 'RDW', 'ALP'], 29990),
                (['Albumin', 'Creatinine', 'MCV'], 25955),
                (['Albumin', 'Creatinine', 'LYM'], 22946)
            ],
            4: [
                (['Albumin', 'Creatinine', 'ALP', 'WBC'], 52168),
                (['Creatinine', 'LYM', 'ALP', 'WBC'], 16068),
                (['Albumin', 'Creatinine', 'MCV', 'ALP'], 15604),
                (['Albumin', 'Creatinine', 'LYM', 'ALP'], 14497),
                (['Albumin', 'Creatinine', 'RDW', 'ALP'], 13333),
                (['Albumin', 'Creatinine', 'LYM', 'WBC'], 12132),
                (['Albumin', 'LYM', 'ALP', 'WBC'], 11056),
                (['Albumin', 'Creatinine', 'Glucose', 'ALP'], 8234),
                (['Creatinine', 'Glucose', 'ALP', 'WBC'], 4181),
                (['Albumin', 'Creatinine', 'Glucose', 'WBC'], 4142)
            ],
            5: [
                (['Albumin', 'Creatinine', 'LYM', 'ALP', 'WBC'], 7691),
                (['Albumin', 'Creatinine', 'Glucose', 'ALP', 'WBC'], 2382),
                (['Albumin', 'Creatinine', 'MCV', 'ALP', 'WBC'], 979),
                (['Albumin', 'Creatinine', 'RDW', 'ALP', 'WBC'], 898),
                (['Albumin', 'Creatinine', 'Glucose', 'MCV', 'ALP'], 653),
                (['Albumin', 'Creatinine', 'Glucose', 'LYM', 'WBC'], 627)
            ]
        }
    
    def generate_synthetic_nhanes_data(self, n_samples=50000):
        """
        Generate synthetic data with realistic correlations based on NHANES patterns
        This is for demonstration - in practice, use real NHANES data
        """
        np.random.seed(42)
        
        # Age distribution (NHANES-like)
        age = np.random.normal(50, 18, n_samples)
        age = np.clip(age, 18, 90)
        
        # Generate correlated biomarkers based on age and health status
        health_status = np.random.normal(0, 1, n_samples) + 0.02 * age
        
        data = pd.DataFrame({
            'age': age,
            'Albumin': np.random.normal(42, 4, n_samples) - 0.05 * age - 0.5 * health_status,
            'Creatinine': np.random.normal(88, 26, n_samples) + 0.3 * age + 2 * health_status,
            'Glucose': np.random.normal(5.6, 2.2, n_samples) + 0.02 * age + 0.3 * health_status,
            'CRP': np.abs(np.random.normal(3, 5, n_samples) + 0.05 * age + 0.8 * health_status),
            'LYM': np.random.normal(30, 8, n_samples) - 0.1 * age - 0.5 * health_status,
            'MCV': np.random.normal(90, 6, n_samples) + 0.05 * age + 0.3 * health_status,
            'RDW': np.random.normal(13.2, 1.3, n_samples) + 0.02 * age + 0.2 * health_status,
            'ALP': np.random.normal(72, 25, n_samples) + 0.2 * age + 1.5 * health_status,
            'WBC': np.random.normal(7.2, 2.1, n_samples) + 0.01 * age + 0.3 * health_status
        })
        
        # Ensure positive values
        for col in data.columns:
            if col != 'age':
                data[col] = np.abs(data[col])
        
        # Add some missing values (realistic pattern)
        for col in ['Glucose', 'LYM', 'RDW', 'MCV']:
            missing_idx = np.random.choice(data.index, size=int(0.1 * len(data)), replace=False)
            data.loc[missing_idx, col] = np.nan
        
        return data
    
    def calculate_full_phenoage(self, data):
        """Calculate PhenoAge using all available biomarkers"""
        # Prepare data
        data_clean = data.copy()
        
        # Log transform CRP
        if 'CRP' in data_clean.columns:
            data_clean['crp_log'] = np.log(data_clean['CRP'] + 0.01)
        
        # Calculate linear predictor
        xb = np.zeros(len(data_clean))
        
        # Add each biomarker's contribution
        for biomarker, weight in self.phenoage_weights.items():
            if biomarker == 'crp_log' and 'crp_log' in data_clean.columns:
                xb += weight * data_clean['crp_log'].fillna(data_clean['crp_log'].median())
            elif biomarker in self.biomarker_map:
                col_name = self.biomarker_map[biomarker]
                if col_name in data_clean.columns:
                    xb += weight * data_clean[col_name].fillna(data_clean[col_name].median())
            elif biomarker == 'age':
                xb += weight * data_clean['age']
        
        # PhenoAge formula (simplified version)
        phenoage = 141.50225 + np.exp(-1.51714 + 0.0054 * xb) / 0.00464
        
        return phenoage
    
    def evaluate_combination(self, data, combination, full_phenoage):
        """Evaluate a specific biomarker combination"""
        # Prepare features
        features = combination + ['age']
        X = data[features].copy()
        
        # Remove rows with missing values
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = full_phenoage[mask]
        
        if len(X) < 100:  # Skip if too few samples
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'n_samples': len(X),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'correlation': np.corrcoef(y_test, y_pred_test)[0, 1],
            'coefficients': dict(zip(features, model.coef_))
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                   scoring='r2', n_jobs=-1)
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        return metrics, model, scaler
    
    def run_validation(self, data=None):
        """Run validation for all datathon combinations"""
        if data is None:
            print("Generating synthetic NHANES-like data...")
            data = self.generate_synthetic_nhanes_data()
        
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Calculate full PhenoAge
        print("\nCalculating full PhenoAge...")
        full_phenoage = self.calculate_full_phenoage(data)
        
        # Results storage
        all_results = []
        
        # Evaluate each combination
        for n_biomarkers, combinations in self.datathon_combinations.items():
            print(f"\nEvaluating {n_biomarkers}-biomarker combinations...")
            
            for combo, user_count in combinations:
                print(f"  Testing: {combo} ({user_count:,} users)")
                
                result = self.evaluate_combination(data, combo, full_phenoage)
                
                if result is not None:
                    metrics, model, scaler = result
                    
                    # Add combination info
                    metrics['n_biomarkers'] = n_biomarkers
                    metrics['combination'] = ', '.join(combo)
                    metrics['datathon_users'] = user_count
                    metrics['biomarkers'] = combo
                    
                    all_results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Calculate efficiency score (performance per biomarker)
        results_df['efficiency_score'] = results_df['test_r2'] / results_df['n_biomarkers']
        
        # Calculate weighted score (considering user count)
        results_df['weighted_score'] = (
            results_df['test_r2'] * np.log10(results_df['datathon_users'] + 1)
        )
        
        return results_df
    
    def create_validation_report(self, results_df):
        """Create comprehensive validation report with visualizations"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Performance by number of biomarkers
        ax1 = plt.subplot(3, 3, 1)
        results_df.boxplot(column='test_r2', by='n_biomarkers', ax=ax1)
        ax1.set_title('Model Performance by Number of Biomarkers')
        ax1.set_xlabel('Number of Biomarkers')
        ax1.set_ylabel('Test R²')
        
        # 2. RMSE comparison
        ax2 = plt.subplot(3, 3, 2)
        results_df.boxplot(column='test_rmse', by='n_biomarkers', ax=ax2)
        ax2.set_title('Prediction Error by Number of Biomarkers')
        ax2.set_xlabel('Number of Biomarkers')
        ax2.set_ylabel('RMSE (years)')
        
        # 3. Top combinations by R²
        ax3 = plt.subplot(3, 3, 3)
        top_10_r2 = results_df.nlargest(10, 'test_r2')
        y_pos = np.arange(len(top_10_r2))
        bars = ax3.barh(y_pos, top_10_r2['test_r2'])
        
        # Color by number of biomarkers
        colors = {3: 'lightblue', 4: 'lightgreen', 5: 'lightcoral'}
        for i, (idx, row) in enumerate(top_10_r2.iterrows()):
            bars[i].set_color(colors[row['n_biomarkers']])
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{row['combination'][:40]}... ({row['n_biomarkers']})" 
                            for _, row in top_10_r2.iterrows()])
        ax3.set_xlabel('Test R²')
        ax3.set_title('Top 10 Combinations by R²')
        ax3.invert_yaxis()
        
        # 4. Efficiency score (R² per biomarker)
        ax4 = plt.subplot(3, 3, 4)
        top_10_eff = results_df.nlargest(10, 'efficiency_score')
        y_pos = np.arange(len(top_10_eff))
        ax4.barh(y_pos, top_10_eff['efficiency_score'])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"{row['combination'][:40]}..." 
                            for _, row in top_10_eff.iterrows()])
        ax4.set_xlabel('Efficiency Score (R² / n_biomarkers)')
        ax4.set_title('Top 10 Most Efficient Combinations')
        ax4.invert_yaxis()
        
        # 5. Weighted score (considering user availability)
        ax5 = plt.subplot(3, 3, 5)
        top_10_weighted = results_df.nlargest(10, 'weighted_score')
        y_pos = np.arange(len(top_10_weighted))
        bars = ax5.barh(y_pos, top_10_weighted['weighted_score'])
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f"{row['combination'][:30]}... ({row['datathon_users']:,})" 
                            for _, row in top_10_weighted.iterrows()])
        ax5.set_xlabel('Weighted Score (R² × log(users))')
        ax5.set_title('Top 10 by Weighted Score (Performance × Availability)')
        ax5.invert_yaxis()
        
        # 6. R² vs User Count scatter
        ax6 = plt.subplot(3, 3, 6)
        for n in [3, 4, 5]:
            subset = results_df[results_df['n_biomarkers'] == n]
            ax6.scatter(subset['datathon_users'], subset['test_r2'], 
                       label=f'{n} biomarkers', s=100, alpha=0.7)
        ax6.set_xscale('log')
        ax6.set_xlabel('Number of Users (log scale)')
        ax6.set_ylabel('Test R²')
        ax6.set_title('Performance vs Availability Trade-off')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Biomarker frequency in top combinations
        ax7 = plt.subplot(3, 3, 7)
        top_20 = results_df.nlargest(20, 'test_r2')
        biomarker_counts = {}
        for _, row in top_20.iterrows():
            for biomarker in row['biomarkers']:
                biomarker_counts[biomarker] = biomarker_counts.get(biomarker, 0) + 1
        
        biomarkers = list(biomarker_counts.keys())
        counts = list(biomarker_counts.values())
        y_pos = np.arange(len(biomarkers))
        ax7.barh(y_pos, counts)
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(biomarkers)
        ax7.set_xlabel('Frequency in Top 20 Combinations')
        ax7.set_title('Most Important Biomarkers')
        ax7.invert_yaxis()
        
        # 8. Cross-validation consistency
        ax8 = plt.subplot(3, 3, 8)
        ax8.scatter(results_df['cv_r2_mean'], results_df['test_r2'], alpha=0.6)
        ax8.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax8.set_xlabel('CV R² (mean)')
        ax8.set_ylabel('Test R²')
        ax8.set_title('Cross-validation Consistency')
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary for each n_biomarkers
        summary_data = []
        for n in [3, 4, 5]:
            subset = results_df[results_df['n_biomarkers'] == n]
            best_combo = subset.loc[subset['test_r2'].idxmax()]
            summary_data.append([
                n,
                f"{best_combo['test_r2']:.3f}",
                f"{best_combo['test_rmse']:.2f}",
                f"{best_combo['datathon_users']:,}",
                best_combo['combination'][:40] + '...'
            ])
        
        table = ax9.table(cellText=summary_data,
                         colLabels=['N Biomarkers', 'Best R²', 'RMSE', 'Users', 'Combination'],
                         cellLoc='left',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax9.set_title('Best Combination per Group', pad=20)
        
        plt.tight_layout()
        plt.savefig('phenoage_validation_results/validation_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results_df.to_csv('phenoage_validation_results/detailed_results.csv', index=False)
        
        # Create recommendation report
        self.create_recommendation_report(results_df)
        
        return results_df
    
    def create_recommendation_report(self, results_df):
        """Create final recommendations based on analysis"""
        report = []
        report.append("=" * 80)
        report.append("PHENOAGE BIOMARKER COMBINATION RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        
        # Overall best performer
        best_overall = results_df.loc[results_df['test_r2'].idxmax()]
        report.append("1. BEST OVERALL PERFORMANCE:")
        report.append(f"   Combination: {best_overall['combination']}")
        report.append(f"   Number of biomarkers: {best_overall['n_biomarkers']}")
        report.append(f"   Test R²: {best_overall['test_r2']:.3f}")
        report.append(f"   RMSE: {best_overall['test_rmse']:.2f} years")
        report.append(f"   Available users: {best_overall['datathon_users']:,}")
        report.append("")
        
        # Best for each group
        report.append("2. BEST COMBINATION PER GROUP:")
        for n in [3, 4, 5]:
            subset = results_df[results_df['n_biomarkers'] == n]
            best = subset.loc[subset['test_r2'].idxmax()]
            report.append(f"\n   {n} Biomarkers:")
            report.append(f"   - Combination: {best['combination']}")
            report.append(f"   - Test R²: {best['test_r2']:.3f}")
            report.append(f"   - RMSE: {best['test_rmse']:.2f} years")
            report.append(f"   - Users: {best['datathon_users']:,}")
        
        # Most practical (good performance with high availability)
        report.append("\n3. MOST PRACTICAL COMBINATIONS (Performance × Availability):")
        top_practical = results_df.nlargest(5, 'weighted_score')
        for i, (_, row) in enumerate(top_practical.iterrows(), 1):
            report.append(f"\n   #{i}: {row['combination']}")
            report.append(f"       R²: {row['test_r2']:.3f}, Users: {row['datathon_users']:,}")
        
        # Key insights
        report.append("\n4. KEY INSIGHTS:")
        
        # Most important biomarkers
        biomarker_importance = {}
        top_quartile = results_df[results_df['test_r2'] > results_df['test_r2'].quantile(0.75)]
        for _, row in top_quartile.iterrows():
            for biomarker in row['biomarkers']:
                if biomarker not in biomarker_importance:
                    biomarker_importance[biomarker] = {'count': 0, 'avg_r2': 0}
                biomarker_importance[biomarker]['count'] += 1
                biomarker_importance[biomarker]['avg_r2'] += row['test_r2']
        
        for biomarker in biomarker_importance:
            biomarker_importance[biomarker]['avg_r2'] /= biomarker_importance[biomarker]['count']
        
        sorted_biomarkers = sorted(biomarker_importance.items(), 
                                  key=lambda x: x[1]['count'], reverse=True)
        
        report.append("\n   Most Important Biomarkers (by frequency in top performers):")
        for biomarker, stats in sorted_biomarkers[:5]:
            report.append(f"   - {biomarker}: appears in {stats['count']} top combinations")
        
        # Performance summary
        report.append("\n   Performance Summary:")
        report.append(f"   - 3 biomarkers: R² range {results_df[results_df['n_biomarkers']==3]['test_r2'].min():.3f} - {results_df[results_df['n_biomarkers']==3]['test_r2'].max():.3f}")
        report.append(f"   - 4 biomarkers: R² range {results_df[results_df['n_biomarkers']==4]['test_r2'].min():.3f} - {results_df[results_df['n_biomarkers']==4]['test_r2'].max():.3f}")
        report.append(f"   - 5 biomarkers: R² range {results_df[results_df['n_biomarkers']==5]['test_r2'].min():.3f} - {results_df[results_df['n_biomarkers']==5]['test_r2'].max():.3f}")
        
        # Final recommendations
        report.append("\n5. RECOMMENDATIONS FOR IMPLEMENTATION:")
        report.append("\n   For Maximum Accuracy:")
        report.append(f"   - Use: {best_overall['combination']}")
        report.append(f"   - Expected R²: {best_overall['test_r2']:.3f}")
        
        report.append("\n   For Maximum Coverage:")
        best_coverage = results_df[results_df['n_biomarkers'] == 3].nlargest(1, 'datathon_users').iloc[0]
        report.append(f"   - Use: {best_coverage['combination']}")
        report.append(f"   - Users: {best_coverage['datathon_users']:,}")
        report.append(f"   - R²: {best_coverage['test_r2']:.3f}")
        
        report.append("\n   Balanced Approach:")
        balanced = results_df[
            (results_df['test_r2'] > results_df['test_r2'].median()) & 
            (results_df['datathon_users'] > 10000)
        ].nlargest(1, 'weighted_score')
        if len(balanced) > 0:
            balanced = balanced.iloc[0]
            report.append(f"   - Use: {balanced['combination']}")
            report.append(f"   - R²: {balanced['test_r2']:.3f}, Users: {balanced['datathon_users']:,}")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        with open('phenoage_validation_results/recommendations.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        
        return report

def main():
    """Run the complete validation analysis"""
    print("Starting PhenoAge Biomarker Combination Validation")
    print("=" * 60)
    
    # Initialize validator
    validator = DatathonPhenoAgeValidator()
    
    # Run validation
    results_df = validator.run_validation()
    
    # Create comprehensive report
    print("\nGenerating validation report...")
    validator.create_validation_report(results_df)
    
    print("\nValidation complete! Check 'phenoage_validation_results' folder for:")
    print("- validation_report.png: Comprehensive visualization")
    print("- detailed_results.csv: Full results data")
    print("- recommendations.txt: Final recommendations")

if __name__ == "__main__":
    main()