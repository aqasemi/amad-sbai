"""
Download and Process NHANES III Data
Uses the correct NHANES III URLs and SAS format files to parse the data
"""

import pandas as pd
import numpy as np
import requests
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('nhanes_iii_data', exist_ok=True)
os.makedirs('nhanes_iii_data/raw', exist_ok=True)

class NHANESIII_Downloader:
    """Downloads and processes NHANES III data"""
    
    def __init__(self):
        self.data_dir = 'nhanes_iii_data'
        self.raw_dir = 'nhanes_iii_data/raw'
        
        # Correct NHANES III data URLs from data.md
        self.urls = {
            'adult_dat': 'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat',
            'adult_sas': 'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.sas',
            'lab_dat': 'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat',
            'lab_sas': 'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.sas'
        }
    
    def download_file(self, url, filename):
        """Download a file from URL"""
        filepath = os.path.join(self.raw_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"{filename} already exists, skipping download")
            return filepath
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"  Saved to {filepath} ({len(response.content) / 1024 / 1024:.1f} MB)")
            return filepath
        except Exception as e:
            print(f"  Error downloading: {e}")
            return None
    
    def parse_sas_format(self, sas_file):
        """Parse SAS format file to extract variable positions"""
        print(f"Parsing SAS format file: {sas_file}")
        
        with open(sas_file, 'r', encoding='latin-1') as f:
            content = f.read()
        
        # Extract INPUT section
        input_section = re.search(r'INPUT\s+(.*?);', content, re.DOTALL | re.IGNORECASE)
        if not input_section:
            print("  Could not find INPUT section in SAS file")
            return {}
        
        input_text = input_section.group(1)
        
        # Parse variable definitions (format: VARNAME $ start-end or VARNAME start-end)
        # Example: SEQN 1-5 or TCP $ 10-15
        variables = {}
        
        # Pattern for: VARNAME [$] start-end [format]
        pattern = r'(\w+)\s+\$?\s+(\d+)\s*-\s*(\d+)'
        
        for match in re.finditer(pattern, input_text):
            varname = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            
            # Convert to 0-based indexing for pandas
            variables[varname] = (start - 1, end)
        
        print(f"  Parsed {len(variables)} variables")
        return variables
    
    def read_fixed_width_file(self, dat_file, sas_file, vars_needed=None):
        """Read fixed-width DAT file using SAS format"""
        print(f"\nReading {os.path.basename(dat_file)}...")
        
        # Parse SAS format
        all_vars = self.parse_sas_format(sas_file)
        
        if not all_vars:
            print("  Warning: No variables parsed from SAS file")
            return None
        
        # Filter to needed variables if specified
        if vars_needed:
            var_positions = {k: v for k, v in all_vars.items() if k in vars_needed}
            print(f"  Extracting {len(var_positions)} needed variables")
        else:
            var_positions = all_vars
        
        if not var_positions:
            print("  Warning: No matching variables found")
            return None
        
        # Create colspecs for pandas
        colspecs = [v for v in var_positions.values()]
        names = list(var_positions.keys())

        # Read the fixed-width file
        df = pd.read_fwf(dat_file, colspecs=colspecs, names=names, 
                         na_values=['.', '', ' '], dtype=str)
        
        print(f"  Loaded {len(df)} records with {len(df.columns)} variables")
        return df
    
    def process_individual_data(self):
        """Download and process individual demographic data (all ages, all columns)"""
        # Download files
        adult_dat = self.download_file(self.urls['adult_dat'], 'adult.dat')
        adult_sas = self.download_file(self.urls['adult_sas'], 'adult.sas')
        
        if not adult_dat or not adult_sas:
            return None
        
        # Read ALL columns (no filtering)
        df = self.read_fixed_width_file(adult_dat, adult_sas, vars_needed=None)
        
        if df is None:
            return None
        
        # Convert columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep all individuals and all columns
        print(f"  Processed {len(df)} individual records with {len(df.columns)} columns (all ages, all features)")
        return df
    
    def process_lab_data(self):
        """Download and process laboratory data (all columns)"""
        # Download files
        lab_dat = self.download_file(self.urls['lab_dat'], 'lab.dat')
        lab_sas = self.download_file(self.urls['lab_sas'], 'lab.sas')
        
        if not lab_dat or not lab_sas:
            return None
        
        # Read ALL columns (no filtering)
        df = self.read_fixed_width_file(lab_dat, lab_sas, vars_needed=None)
        
        if df is None:
            return None
        
        # Convert columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"  Processed {len(df)} lab records with {len(df.columns)} columns (all features)")
        return df
    
    def combine_all_data(self):
        """Combine all data sources"""
        print("\n" + "=" * 60)
        print("COMBINING ALL DATA SOURCES")
        print("=" * 60)
        
        # Process each data source
        individual_df = self.process_individual_data()
        lab_df = self.process_lab_data()
        
        if individual_df is None:
            print("ERROR: Could not load individual data")
            return None
        
        print(f"\nIndividual data: {len(individual_df)} records, {len(individual_df.columns)} columns")
        
        # Start with individual demographics
        combined = individual_df.copy()
        
        # Merge lab data on SEQN
        if lab_df is not None:
            print(f"Lab data: {len(lab_df)} records, {len(lab_df.columns)} columns")
            
            # Merge on SEQN (it exists in both files)
            combined = combined.merge(lab_df, on='SEQN', how='left', suffixes=('', '_lab'))
            print(f"\nAfter merging labs: {len(combined)} records, {len(combined.columns)} columns")
        
        return combined
    
    def create_analysis_datasets(self):
        """Create analysis-ready datasets"""
        # Combine all data
        full_data = self.combine_all_data()
        
        if full_data is None:
            print("\nERROR: Could not create combined dataset")
            return
        
        # Save full dataset with all columns
        full_path = os.path.join(self.data_dir, 'nhanes_iii_full.csv')
        full_data.to_csv(full_path, index=False)
        print(f"\n✓ Saved full dataset: {full_path}")
        print(f"  Total records: {len(full_data):,}")
        print(f"  Total columns: {len(full_data.columns):,}")
        
        # Print column summary
        print("\n" + "=" * 60)
        print("COLUMN SUMMARY")
        print("=" * 60)
        
        # Show first 50 columns as preview
        print(f"\nFirst 50 columns (out of {len(full_data.columns)}):")
        for i, col in enumerate(full_data.columns[:50], 1):
            n_available = full_data[col].notna().sum()
            pct = n_available / len(full_data) * 100
            print(f"{i:3d}. {col:20s}: {n_available:7,} non-null ({pct:5.1f}%)")
        
        if len(full_data.columns) > 50:
            print(f"\n... and {len(full_data.columns) - 50} more columns")
        
        # Save column info to CSV
        column_info = []
        for col in full_data.columns:
            n_available = full_data[col].notna().sum()
            n_missing = full_data[col].isna().sum()
            pct_available = n_available / len(full_data) * 100
            
            column_info.append({
                'column_name': col,
                'non_null_count': n_available,
                'null_count': n_missing,
                'percent_available': pct_available,
                'dtype': str(full_data[col].dtype)
            })
        
        column_info_df = pd.DataFrame(column_info)
        column_info_path = os.path.join(self.data_dir, 'nhanes_iii_column_info.csv')
        column_info_df.to_csv(column_info_path, index=False)
        print(f"\n✓ Saved column information: {column_info_path}")
        
        return full_data

def main():
    """Main function"""
    print("=" * 60)
    print("NHANES III DATA DOWNLOAD AND PROCESSING")
    print("=" * 60)
    
    downloader = NHANESIII_Downloader()
    
    try:
        full_data = downloader.create_analysis_datasets()
        
        if full_data is not None:
            print("\n" + "=" * 60)
            print("✓ DOWNLOAD AND PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"\nData saved in: {downloader.data_dir}/")
        else:
            print("\n✗ Failed to create datasets")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
