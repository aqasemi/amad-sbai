"""
NHANES Data Download and Preparation Script
Downloads NHANES III and IV data with the specific biomarkers needed for PhenoAge analysis
"""

import pandas as pd
import numpy as np
import requests
import os
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Create data directory
os.makedirs('nhanes_data', exist_ok=True)

class NHANESDataDownloader:
    """Downloads and prepares NHANES data for PhenoAge analysis"""
    
    def __init__(self):
        self.base_url = "https://wwwn.cdc.gov/Nchs/Nhanes"
        self.data_dir = "nhanes_data"
        
        # Define the 9 PhenoAge biomarkers
        self.phenoage_biomarkers = {
            'albumin': 'LBXSAL',  # Albumin (g/L)
            'creatinine': 'LBXSCR',  # Creatinine (umol/L) 
            'glucose': 'LBXGLU',  # Glucose, serum (mmol/L)
            'crp': 'LBXCRP',  # C-reactive protein (mg/dL)
            'lymphocyte_pct': 'LBXLYPCT',  # Lymphocyte percent (%)
            'mcv': 'LBXMCVSI',  # Mean cell volume (fL)
            'rdw': 'LBXRDW',  # Red cell distribution width (%)
            'alkaline_phosphatase': 'LBXSAPSI',  # Alkaline phosphatase (U/L)
            'wbc': 'LBXWBCSI'  # White blood cell count (1000 cells/uL)
        }
        
    def download_xpt_file(self, url, filename):
        """Download XPT file from NHANES"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save the file
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Read the XPT file
            df = pd.read_sas(filepath, format='xport')
            return df
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def get_nhanes_iii_data(self):
        """Download NHANES III data (1988-1994)"""
        print("Downloading NHANES III data...")
        
        # NHANES III uses different file structure
        # We'll need to combine multiple files
        
        # Demographics
        demo_url = f"{self.base_url}/1999-2000/DEMO.XPT"
        demo_df = self.download_xpt_file(demo_url, "nhanes3_demo.xpt")
        
        # Lab data - Standard Biochemistry Profile
        lab_url = f"{self.base_url}/1999-2000/LAB18.XPT"
        lab_df = self.download_xpt_file(lab_url, "nhanes3_lab.xpt")
        
        # CBC data
        cbc_url = f"{self.base_url}/1999-2000/LAB25.XPT"
        cbc_df = self.download_xpt_file(cbc_url, "nhanes3_cbc.xpt")
        
        # Mortality data
        mort_url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_III_MORT_2019_PUBLIC.dat"
        
        # Note: For this example, we'll use a simplified approach
        # In practice, you'd need to properly parse the mortality file
        
        return demo_df, lab_df, cbc_df
    
    def get_nhanes_continuous_data(self, years):
        """Download NHANES continuous data for specified years"""
        print(f"Downloading NHANES {years} data...")
        
        dfs = {}
        
        # Demographics
        demo_url = f"{self.base_url}/{years}/DEMO_{years[-1]}.XPT"
        dfs['demo'] = self.download_xpt_file(demo_url, f"nhanes_{years}_demo.xpt")
        
        # Standard Biochemistry Profile
        biopro_url = f"{self.base_url}/{years}/BIOPRO_{years[-1]}.XPT"
        dfs['biopro'] = self.download_xpt_file(biopro_url, f"nhanes_{years}_biopro.xpt")
        
        # CBC
        cbc_url = f"{self.base_url}/{years}/CBC_{years[-1]}.XPT"
        dfs['cbc'] = self.download_xpt_file(cbc_url, f"nhanes_{years}_cbc.xpt")
        
        # CRP
        crp_url = f"{self.base_url}/{years}/CRP_{years[-1]}.XPT"
        dfs['crp'] = self.download_xpt_file(crp_url, f"nhanes_{years}_crp.xpt")
        
        # Glucose
        glu_url = f"{self.base_url}/{years}/GLU_{years[-1]}.XPT"
        dfs['glucose'] = self.download_xpt_file(glu_url, f"nhanes_{years}_glucose.xpt")
        
        return dfs
    
    def prepare_phenoage_dataset(self, data_dfs):
        """Prepare dataset with PhenoAge biomarkers"""
        print("Preparing PhenoAge dataset...")
        
        # Start with demographics
        if 'demo' in data_dfs and data_dfs['demo'] is not None:
            df = data_dfs['demo'][['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1']].copy()
            df.columns = ['SEQN', 'age', 'gender', 'race']
            
            # Merge other datasets
            for key, data_df in data_dfs.items():
                if key != 'demo' and data_df is not None:
                    # Get relevant columns
                    cols_to_keep = ['SEQN'] + [col for col in data_df.columns if col in self.phenoage_biomarkers.values()]
                    if len(cols_to_keep) > 1:
                        df = df.merge(data_df[cols_to_keep], on='SEQN', how='left')
            
            # Rename columns to friendly names
            rename_dict = {v: k for k, v in self.phenoage_biomarkers.items()}
            df.rename(columns=rename_dict, inplace=True)
            
            return df
        
        return None

def main():
    """Main function to download and prepare NHANES data"""
    downloader = NHANESDataDownloader()
    
    # Download NHANES continuous data (we'll use 2015-2016 as an example)
    # In practice, you'd want multiple years
    years = "2015-2016"
    data_dfs = downloader.get_nhanes_continuous_data(years)
    
    # Prepare the dataset
    phenoage_df = downloader.prepare_phenoage_dataset(data_dfs)
    
    if phenoage_df is not None:
        # Save the prepared dataset
        phenoage_df.to_csv('nhanes_data/nhanes_phenoage_data.csv', index=False)
        print(f"Saved PhenoAge dataset with {len(phenoage_df)} participants")
        print(f"Columns: {list(phenoage_df.columns)}")
        print(f"Missing values:\n{phenoage_df.isnull().sum()}")
    else:
        print("Failed to prepare PhenoAge dataset")
    
    # Create a summary of available biomarkers
    if phenoage_df is not None:
        biomarker_cols = [col for col in phenoage_df.columns if col in downloader.phenoage_biomarkers.keys()]
        summary = []
        for col in biomarker_cols:
            n_available = phenoage_df[col].notna().sum()
            pct_available = (n_available / len(phenoage_df)) * 100
            summary.append({
                'biomarker': col,
                'n_available': n_available,
                'pct_available': pct_available
            })
        
        summary_df = pd.DataFrame(summary).sort_values('n_available', ascending=False)
        summary_df.to_csv('nhanes_data/biomarker_availability_summary.csv', index=False)
        print("\nBiomarker availability summary:")
        print(summary_df)

if __name__ == "__main__":
    main()