"""
PhenoAge Implementation and Validation
Implements the Levine PhenoAge methodology and tests different biomarker combinations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import ElasticNetCV, CoxPHSurvivalAnalysis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class PhenoAgeCalculator:
    """Implements the PhenoAge calculation methodology"""
    
    def __init__(self):
        # PhenoAge coefficients from the paper (Table 1)
        self.phenoage_coefficients = {
            'albumin': -0.0336,
            'creatinine': 0.0095,
            'glucose': 0.1953,
            'crp_log': 0.0954,  # Note: CRP should be log-transformed
            'lymphocyte_pct': -0.0120,
            'mcv': 0.0268,
            'rdw': 0.3306,
            'alkaline_phosphatase': 0.0019,
            'wbc': 0.0554,
            'age': 0.0804
        }
        
        # Gompertz parameters from the paper
        self.gamma = 0.092  # Gompertz gamma parameter
        self.lambda0 = 0.00002193  # Baseline hazard
        
    def calculate_phenoage(self, data, biomarkers=None):
        """
        Calculate PhenoAge using the published formula
        
        Parameters:
        -----------
        data : DataFrame
            Data containing biomarkers and age
        biomarkers : list
            List of biomarkers to use (if None, uses all available)
            
        Returns:
        --------
        phenoage : array
            Calculated PhenoAge values
        """
        if biomarkers is None:
            biomarkers = list(self.phenoage_coefficients.keys())
        
        # Initialize linear predictor
        xb = 0
        
        # Add contributions from each biomarker
        for biomarker in biomarkers:
            if biomarker in data.columns and biomarker in self.phenoage_coefficients:
                # Handle CRP log transformation
                if biomarker == 'crp':
                    values = np.log(data[biomarker] + 0.01)  # Add small constant to avoid log(0)
                    xb += self.phenoage_coefficients['crp_log'] * values
                else:
                    xb += self.phenoage_coefficients[biomarker] * data[biomarker]
        
        # Calculate mortality risk
        mortality_risk = 1 - np.exp(-self.lambda0 * np.exp(xb))
        
        # Convert to PhenoAge (in years)
        phenoage = -np.log(1 - mortality_risk) / self.gamma
        
        return phenoage
    
    def train_reduced_model(self, X, y, biomarker_subset):
        """
        Train a reduced model using only a subset of biomarkers
        
        Parameters:
        -----------
        X : DataFrame
            Full dataset with all biomarkers
        y : array
            Target PhenoAge values (calculated from full model)
        biomarker_subset : list
            List of biomarkers to use in the reduced model
            
        Returns:
        --------
        model : fitted model
        metrics : dict of performance metrics
        """
        # Select subset of features
        X_subset = X[biomarker_subset + ['age']].copy()
        
        # Handle missing values
        X_subset = X_subset.dropna()
        y_subset = y[X_subset.index]
        
        # Train ElasticNet model with cross-validation
        model = ElasticNetCV(cv=10, random_state=42, max_iter=10000)
        model.fit(X_subset, y_subset)
        
        # Get predictions
        y_pred = cross_val_predict(model, X_subset, y_subset, cv=10)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_subset, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_subset, y_pred)),
            'mae': mean_absolute_error(y_subset, y_pred),
            'correlation': np.corrcoef(y_subset, y_pred)[0, 1],
            'n_samples': len(y_subset)
        }
        
        return model, metrics

class BiomarkerCombinationAnalyzer:
    """Analyzes different combinations of biomarkers for PhenoAge calculation"""
    
    def __init__(self, data):
        self.data = data
        self.calculator = PhenoAgeCalculator()
        self.results = []
        
    def prepare_data(self):
        """Prepare data for analysis"""
        # List of all PhenoAge biomarkers
        all_biomarkers = ['albumin', 'creatinine', 'glucose', 'crp', 
                         'lymphocyte_pct', 'mcv', 'rdw', 
                         'alkaline_phosphatase', 'wbc']
        
        # Check which biomarkers are available in the data
        available_biomarkers = [b for b in all_biomarkers if b in self.data.columns]
        
        print(f"Available biomarkers: {available_biomarkers}")
        
        # Calculate full PhenoAge (with all available biomarkers) as ground truth
        self.full_phenoage = self.calculator.calculate_phenoage(self.data, available_biomarkers)
        
        return available_biomarkers
    
    def analyze_combinations(self, n_biomarkers, available_biomarkers):
        """
        Analyze all combinations of n biomarkers
        
        Parameters:
        -----------
        n_biomarkers : int
            Number of biomarkers in each combination
        available_biomarkers : list
            List of available biomarkers
        """
        print(f"\nAnalyzing {n_biomarkers}-biomarker combinations...")
        
        # Generate all combinations
        biomarker_combinations = list(combinations(available_biomarkers, n_biomarkers))
        
        for combo in biomarker_combinations:
            combo_list = list(combo)
            
            # Train reduced model
            try:
                model, metrics = self.calculator.train_reduced_model(
                    self.data, 
                    self.full_phenoage, 
                    combo_list
                )
                
                # Store results
                result = {
                    'n_biomarkers': n_biomarkers,
                    'biomarkers': ', '.join(combo_list),
                    'biomarker_list': combo_list,
                    **metrics
                }
                
                self.results.append(result)
                
            except Exception as e:
                print(f"Error with combination {combo_list}: {e}")
    
    def get_best_combinations(self, n_top=10):
        """Get the best performing combinations"""
        results_df = pd.DataFrame(self.results)
        
        # Sort by R² score (you could also use RMSE or MAE)
        results_df = results_df.sort_values('r2', ascending=False)
        
        return results_df.head(n_top)
    
    def visualize_results(self, save_path='phenoage_analysis_results.png'):
        """Create visualizations of the results"""
        results_df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. R² by number of biomarkers
        ax = axes[0, 0]
        results_df.boxplot(column='r2', by='n_biomarkers', ax=ax)
        ax.set_title('R² Score by Number of Biomarkers')
        ax.set_xlabel('Number of Biomarkers')
        ax.set_ylabel('R² Score')
        
        # 2. RMSE by number of biomarkers
        ax = axes[0, 1]
        results_df.boxplot(column='rmse', by='n_biomarkers', ax=ax)
        ax.set_title('RMSE by Number of Biomarkers')
        ax.set_xlabel('Number of Biomarkers')
        ax.set_ylabel('RMSE (years)')
        
        # 3. Top 10 combinations by R²
        ax = axes[1, 0]
        top_10 = self.get_best_combinations(10)
        y_pos = np.arange(len(top_10))
        ax.barh(y_pos, top_10['r2'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([b[:30] + '...' if len(b) > 30 else b for b in top_10['biomarkers']])
        ax.set_xlabel('R² Score')
        ax.set_title('Top 10 Biomarker Combinations by R²')
        ax.invert_yaxis()
        
        # 4. Sample size distribution
        ax = axes[1, 1]
        results_df['n_samples'].hist(bins=30, ax=ax)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Frequency')
        ax.set_title('Sample Size Distribution Across Combinations')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return results_df

def analyze_datathon_combinations():
    """Analyze the specific combinations available in the datathon dataset"""
    
    # Define the combinations from the plan
    datathon_combinations = {
        3: [
            ['albumin', 'creatinine', 'alkaline_phosphatase'],  # 209757 users
            ['creatinine', 'alkaline_phosphatase', 'wbc'],      # 112089 users
            ['albumin', 'creatinine', 'wbc'],                   # 83543 users
            ['albumin', 'alkaline_phosphatase', 'wbc'],         # 75431 users
            ['creatinine', 'lymphocyte_pct', 'wbc'],           # 46978 users
        ],
        4: [
            ['albumin', 'creatinine', 'alkaline_phosphatase', 'wbc'],     # 52168 users
            ['creatinine', 'lymphocyte_pct', 'alkaline_phosphatase', 'wbc'],  # 16068 users
            ['albumin', 'creatinine', 'mcv', 'alkaline_phosphatase'],     # 15604 users
            ['albumin', 'creatinine', 'lymphocyte_pct', 'alkaline_phosphatase'],  # 14497 users
            ['albumin', 'creatinine', 'rdw', 'alkaline_phosphatase'],     # 13333 users
        ],
        5: [
            ['albumin', 'creatinine', 'lymphocyte_pct', 'alkaline_phosphatase', 'wbc'],  # 7691 users
            ['albumin', 'creatinine', 'glucose', 'alkaline_phosphatase', 'wbc'],  # 2382 users
            ['albumin', 'creatinine', 'mcv', 'alkaline_phosphatase', 'wbc'],      # 979 users
            ['albumin', 'creatinine', 'rdw', 'alkaline_phosphatase', 'wbc'],      # 898 users
            ['albumin', 'creatinine', 'glucose', 'mcv', 'alkaline_phosphatase'],   # 653 users
        ]
    }
    
    return datathon_combinations

def main():
    """Main analysis function"""
    print("PhenoAge Biomarker Combination Analysis")
    print("=" * 50)
    
    # Load or generate sample data
    # In practice, you would load the NHANES data here
    # For demonstration, we'll create synthetic data
    
    np.random.seed(42)
    n_samples = 10000
    
    # Create synthetic dataset with correlations similar to real biomarkers
    data = pd.DataFrame({
        'age': np.random.normal(50, 15, n_samples),
        'albumin': np.random.normal(40, 5, n_samples),
        'creatinine': np.random.normal(88, 20, n_samples),
        'glucose': np.random.normal(5.5, 1.5, n_samples),
        'crp': np.random.lognormal(0.5, 1, n_samples),
        'lymphocyte_pct': np.random.normal(30, 8, n_samples),
        'mcv': np.random.normal(90, 5, n_samples),
        'rdw': np.random.normal(13, 1.5, n_samples),
        'alkaline_phosphatase': np.random.normal(70, 20, n_samples),
        'wbc': np.random.normal(7, 2, n_samples)
    })
    
    # Add some realistic correlations
    data['albumin'] = data['albumin'] - 0.1 * data['age']
    data['creatinine'] = data['creatinine'] + 0.3 * data['age']
    data['glucose'] = data['glucose'] + 0.02 * data['age']
    
    # Initialize analyzer
    analyzer = BiomarkerCombinationAnalyzer(data)
    available_biomarkers = analyzer.prepare_data()
    
    # Analyze combinations of 3, 4, and 5 biomarkers
    for n in [3, 4, 5]:
        analyzer.analyze_combinations(n, available_biomarkers)
    
    # Visualize results
    results_df = analyzer.visualize_results()
    
    # Save detailed results
    results_df.to_csv('phenoage_combination_results.csv', index=False)
    
    # Get best combinations for each group size
    print("\nBest combinations by number of biomarkers:")
    for n in [3, 4, 5]:
        print(f"\n{n} biomarkers:")
        best = results_df[results_df['n_biomarkers'] == n].head(5)
        for _, row in best.iterrows():
            print(f"  {row['biomarkers']}: R²={row['r2']:.3f}, RMSE={row['rmse']:.2f}")
    
    # Analyze specific datathon combinations
    datathon_combos = analyze_datathon_combinations()
    
    print("\n\nAnalyzing specific datathon combinations:")
    datathon_results = []
    
    for n_biomarkers, combos in datathon_combos.items():
        for combo in combos:
            try:
                model, metrics = analyzer.calculator.train_reduced_model(
                    data, 
                    analyzer.full_phenoage, 
                    combo
                )
                
                result = {
                    'n_biomarkers': n_biomarkers,
                    'biomarkers': ', '.join(combo),
                    **metrics
                }
                datathon_results.append(result)
                
            except Exception as e:
                print(f"Error with combination {combo}: {e}")
    
    # Save datathon-specific results
    datathon_df = pd.DataFrame(datathon_results)
    datathon_df = datathon_df.sort_values('r2', ascending=False)
    datathon_df.to_csv('datathon_combination_results.csv', index=False)
    
    print("\nTop datathon combinations:")
    print(datathon_df.head(10))

if __name__ == "__main__":
    main()