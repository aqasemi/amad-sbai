import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def digi_cot(data_mat):
    cot = data_mat['LBXCOT'].copy()
    data_mat.loc[cot < 10, 'LBXCOT'] = 0
    data_mat.loc[(cot >= 10) & (cot < 100), 'LBXCOT'] = 1
    data_mat.loc[(cot >= 100) & (cot < 200), 'LBXCOT'] = 2
    data_mat.loc[cot >= 200, 'LBXCOT'] = 3
    return data_mat

def pop_pc_fi_fs1(q_data_mat):
    bpq020 = q_data_mat['BPQ020'].fillna(2)
    diq010 = q_data_mat['DIQ010'].fillna(2)
    huq010 = q_data_mat['HUQ010'].fillna(3)
    huq020 = q_data_mat['HUQ020'].fillna(3)
    huq050 = q_data_mat['HUQ050'].fillna(0)
    huq070 = q_data_mat['HUQ070'].fillna(2)
    kiq020 = q_data_mat['KIQ020'].fillna(2)
    mcq010 = q_data_mat['MCQ010'].fillna(2)
    mcq053 = q_data_mat['MCQ053'].fillna(2)
    mcq160a = q_data_mat['MCQ160A'].fillna(2)
    mcq160b = q_data_mat['MCQ160B'].fillna(2)
    mcq160c = q_data_mat['MCQ160C'].fillna(2)
    mcq160d = q_data_mat['MCQ160D'].fillna(2)
    mcq160e = q_data_mat['MCQ160E'].fillna(2)
    mcq160f = q_data_mat['MCQ160F'].fillna(2)
    mcq160g = q_data_mat['MCQ160G'].fillna(2)
    mcq160i = q_data_mat['MCQ160I'].fillna(2)
    mcq160j = q_data_mat['MCQ160J'].fillna(2)
    mcq160k = q_data_mat['MCQ160K'].fillna(2)
    mcq160l = q_data_mat['MCQ160L'].fillna(2)
    mcq220 = q_data_mat['MCQ220'].fillna(2)
    osq010a = q_data_mat['OSQ010A'].fillna(2)
    osq010b = q_data_mat['OSQ010B'].fillna(2)
    osq010c = q_data_mat['OSQ010C'].fillna(2)
    osq060 = q_data_mat['OSQ060'].fillna(2)
    pfq056 = q_data_mat['PFQ056'].fillna(2)

    bin_vec = np.column_stack([
        (bpq020 == 1),
        ((diq010 == 1) | (diq010 == 3)),
        (kiq020 == 1),
        (mcq010 == 1),
        (mcq053 == 1),
        (mcq160a == 1),
        (mcq160b == 1),
        (mcq160c == 1),
        (mcq160d == 1),
        (mcq160e == 1),
        (mcq160f == 1),
        (mcq160g == 1),
        (mcq160i == 1),
        (mcq160j == 1),
        (mcq160k == 1),
        (mcq160l == 1),
        (mcq220 == 1),
        (osq010a == 1),
        (osq010b == 1),
        (osq010c == 1),
        (osq060 == 1),
        (pfq056 == 1),
        (huq070 == 1)
    ]).astype(int)

    sum_over_bin_vec = bin_vec.sum(axis=1) / 22.0
    return sum_over_bin_vec

def pop_pc_fi_fs2(q_data_mat):
    huq010 = q_data_mat['HUQ010'].fillna(3)
    huq020 = q_data_mat['HUQ020'].fillna(3)
    huq050 = q_data_mat['HUQ050'].fillna(0)
    huq070 = q_data_mat['HUQ070'].fillna(2)

    a_vec = (huq010 == 4).astype(int) * 2 + (huq010 == 5).astype(int) * 4
    d_vec = 1 - (huq020 == 1).astype(float) * 0.5 + (huq020 == 2).astype(int)
    f_score = a_vec * d_vec
    return f_score

def pop_pc_fi_fs3(q_data_mat):
    huq050 = q_data_mat['HUQ050'].fillna(0)
    huq050 = huq050.replace({77: 0, 99: 0})
    return huq050

def populate_ldl(data_mat, q_data_mat):
    tot_cv = data_mat['LBDTCSI']
    hdl_v = data_mat['LBDHDLSI']
    tri_gv = data_mat['LBDSTRSI']
    mask = ~(tot_cv.isna() | hdl_v.isna() | tri_gv.isna())
    ldl_vec = np.zeros(len(data_mat))
    ldl_vec[mask] = (tot_cv[mask] - (tri_gv[mask] / 5) - hdl_v[mask])
    return ldl_vec

def pop_cr_alb_rat(data_mat):
    crea_vals = data_mat['URXUCRSI']
    albu_vals = data_mat['URXUMASI']
    cr_alb_rat = albu_vals / (crea_vals * 1.1312e-4)
    return cr_alb_rat

def pop_lin_age(lin_model, data_mat, q_data_mat):
    data_flags = lin_model['parType']
    sex_flag = lin_model['sexFlag']
    data_pars = lin_model.loc[(data_flags == 'DATA') & (sex_flag == 1), 'parName'].values

    # Extract model parameters
    beta_m = lin_model.loc[sex_flag == 1, 'betaVal'].values
    n_paras = len(beta_m) - 2
    beta0_m = beta_m[n_paras]
    c1_m = beta_m[n_paras + 1]
    beta_m = beta_m[:n_paras]

    beta_f = lin_model.loc[sex_flag == 2, 'betaVal'].values
    n_paras = len(beta_f) - 2
    beta0_f = beta_f[n_paras]
    c1_f = beta_f[n_paras + 1]
    beta_f = beta_f[:n_paras]

    med_vec_m = lin_model.loc[(data_flags == 'DATA') & (sex_flag == 1), 'medVal'].values
    mad_vec_m = lin_model.loc[(data_flags == 'DATA') & (sex_flag == 1), 'madVal'].values
    med_vec_m[np.isnan(med_vec_m)] = 0
    mad_vec_m[np.isnan(mad_vec_m)] = 1

    med_vec_f = lin_model.loc[(data_flags == 'DATA') & (sex_flag == 2), 'medVal'].values
    mad_vec_f = lin_model.loc[(data_flags == 'DATA') & (sex_flag == 2), 'madVal'].values
    med_vec_f[np.isnan(med_vec_f)] = 0
    mad_vec_f[np.isnan(mad_vec_f)] = 1

    n_seqs = len(data_mat)
    del_age_vec = np.zeros(n_seqs)

    for i in range(n_seqs):
        next_sex = q_data_mat.loc[i, 'RIAGENDR']
        next_age = q_data_mat.loc[i, 'RIDAGEYR']
        next_dat_vec = data_mat.loc[i, data_pars].values.astype(float)

        if next_sex == 1:
            next_dat_vec = (next_dat_vec - med_vec_m) / mad_vec_m
            par_x_beta = next_dat_vec * beta_m
            del_age = c1_m + beta0_m * next_age * 12 + np.sum(par_x_beta)
        elif next_sex == 2:
            next_dat_vec = (next_dat_vec - med_vec_f) / mad_vec_f
            par_x_beta = next_dat_vec * beta_f
            del_age = c1_f + beta0_f * next_age * 12 + np.sum(par_x_beta)
        else:
            del_age = 0  # or handle

        del_age_vec[i] = del_age

    return del_age_vec

def calculate_pheno_age(data_mat, q_data_mat):
    def compute_single(biomarkers):
        # Extract values
        age = biomarkers['age']
        albumin = biomarkers['albumin']  # already in g/dL
        creatinine = biomarkers['creatinine']  # already in mg/dL
        glucose = biomarkers['glucose']
        crp = biomarkers['crp']  # already in mg/L
        lymphocyte_percent = biomarkers['lymphocytePercent']
        mean_cell_volume = biomarkers['meanCellVolume']
        red_cell_dist_width = biomarkers['redCellDistWidth']
        alkaline_phosphatase = biomarkers['alkalinePhosphatase']
        white_blood_cell_count = biomarkers['whiteBloodCellCount']

        # Calculate xb
        xb = (-19.907 -
              0.0336 * albumin +
              0.0095 * creatinine +
              0.1953 * glucose +
              0.0954 * np.log(crp) -
              0.0120 * lymphocyte_percent +
              0.0268 * mean_cell_volume +
              0.3306 * red_cell_dist_width +
              0.00188 * alkaline_phosphatase +
              0.0554 * white_blood_cell_count +
              0.0804 * age)

        # Calculate mortality score
        gamma = 0.0076927
        exp_xb = np.exp(xb)
        mortality_score = 1 - np.exp(-exp_xb * (np.exp(120 * gamma) - 1) / gamma)

        # Clamp mortality score
        max_mortality = 0.999999
        min_mortality = 0.000001
        original_mortality = mortality_score
        mortality_score = np.clip(mortality_score, min_mortality, max_mortality)
        was_clamped = mortality_score != original_mortality

        # Calculate PhenoAge
        try:
            pheno_age = 141.50225 + (np.log(-0.00553 * np.log(1 - mortality_score)) / 0.090165)
            if not np.isfinite(pheno_age):
                return np.nan
        except:
            return np.nan

        return pheno_age

    # Prepare biomarkers DataFrame with unit conversions
    biomarkers_df = pd.DataFrame({
        'age': q_data_mat['RIDAGEYR'],
        'albumin': data_mat['LBDSALSI'],  # g/L -> g/dL
        'creatinine': data_mat['LBDSCRSI'] ,  # umol/L -> mg/dL
        'glucose': data_mat['LBDSGLSI'],  # keep mmol/L to avoid mortality saturation
        'crp': (original_crp * 10).clip(lower=0.01),  # mg/dL -> mg/L
        'lymphocytePercent': data_mat['LBXLYPCT'],
        'meanCellVolume': data_mat['LBXMCVSI'],
        'redCellDistWidth': data_mat['LBXRDW'],
        'alkalinePhosphatase': data_mat['LBXSAPSI'],
        'whiteBloodCellCount': data_mat['LBXWBCSI']
    })

    # Apply computation row-wise
    pheno_ages = biomarkers_df.apply(compute_single, axis=1)
    print(pheno_ages.head(5))
    print(biomarkers_df.head(5).to_string())
    return pheno_ages

def calculate_pheno_mortality_score(data_mat, q_data_mat):
    def compute_mortality(biomarkers):
        age = biomarkers['age']
        albumin = biomarkers['albumin']
        creatinine = biomarkers['creatinine']
        glucose = biomarkers['glucose']
        crp = biomarkers['crp']
        lymphocyte_percent = biomarkers['lymphocytePercent']
        mean_cell_volume = biomarkers['meanCellVolume']
        red_cell_dist_width = biomarkers['redCellDistWidth']
        alkaline_phosphatase = biomarkers['alkalinePhosphatase']
        white_blood_cell_count = biomarkers['whiteBloodCellCount']

        xb = (-19.907 -
              0.0336 * albumin +
              0.0095 * creatinine +
              0.1953 * glucose +
              0.0954 * np.log(crp) -
              0.0120 * lymphocyte_percent +
              0.0268 * mean_cell_volume +
              0.3306 * red_cell_dist_width +
              0.00188 * alkaline_phosphatase +
              0.0554 * white_blood_cell_count +
              0.0804 * age)

        gamma = 0.0076927
        exp_xb = np.exp(xb)
        mortality_score = 1 - np.exp(-exp_xb * (np.exp(120 * gamma) - 1) / gamma)
        mortality_score = np.clip(mortality_score, 0.000001, 0.999999)
        return mortality_score

    biomarkers_df = pd.DataFrame({
        'age': q_data_mat['RIDAGEYR'],
        'albumin': data_mat['LBDSALSI'],
        'creatinine': data_mat['LBDSCRSI'],
        'glucose': data_mat['LBDSGLSI'],
        'crp': (original_crp * 10).clip(lower=0.01),
        'lymphocytePercent': data_mat['LBXLYPCT'],
        'meanCellVolume': data_mat['LBXMCVSI'],
        'redCellDistWidth': data_mat['LBXRDW'],
        'alkalinePhosphatase': data_mat['LBXSAPSI'],
        'whiteBloodCellCount': data_mat['LBXWBCSI']
    })

    # print first 10 rows of biomarkers along with the calculated mortality score
    print(biomarkers_df.head(10))
    print(biomarkers_df.apply(compute_mortality, axis=1).head(10))

    return biomarkers_df.apply(compute_mortality, axis=1)

def assign_age_bin_label(age_value, bin_width=10):
    if pd.isna(age_value):
        return np.nan
    start = int(np.floor(age_value / bin_width) * bin_width)
    end = start + bin_width - 1
    return f"{start}-{end}"

def compute_bai(data_mat, q_data_mat, bin_width=10):
    # Prepare grouping keys
    age_series = data_mat['chronAge']
    sex_series = q_data_mat['RIAGENDR']
    data_mat['ageBin'] = age_series.apply(lambda a: assign_age_bin_label(a, bin_width))
    data_mat['sex'] = sex_series.map({1: 'Male', 2: 'Female'})

    # Reference stats from current cohort (can be replaced with external reference later)
    ref = data_mat[['ageBin', 'sex', 'delAge']].copy()
    stats = (ref.groupby(['sex', 'ageBin'])['delAge']
                 .agg(['mean', 'std'])
                 .reset_index()
                 .rename(columns={'mean': 'bai_mean', 'std': 'bai_std'}))
    stats['bai_std'] = stats['bai_std'].replace(0, np.nan)

    # Merge stats into main frame
    data_mat = data_mat.merge(stats, on=['sex', 'ageBin'], how='left')

    # Compute BAI z-score
    data_mat['BAI'] = (data_mat['delAge'] - data_mat['bai_mean']) / data_mat['bai_std']
    data_mat['BAI'] = data_mat['BAI'].replace([np.inf, -np.inf], np.nan)

    # Categorize BAI
    def categorize_bai(val):
        if pd.isna(val):
            return 'Unknown'
        if val > 2.0:
            return 'Highly Accelerated'
        if 1.0 < val <= 2.0:
            return 'Accelerated'
        if -1.0 <= val <= 1.0:
            return 'Normal Aging'
        return 'Decelerated'

    data_mat['BAICategory'] = data_mat['BAI'].apply(categorize_bai)
    return data_mat

def run_linage_on_frames(data_mat: pd.DataFrame, q_data_mat: pd.DataFrame, lin_age_pars_path: str,
                         compute_bai_within_sample: bool = True, bin_width: int = 10) -> pd.DataFrame:
    """Run the linAge pipeline on provided data frames and return an augmented data frame.

    data_mat: biomarker/clinical parameters (NHANES variable names)
    q_data_mat: questionnaire/demographics with at least RIAGENDR, RIDAGEYR
    lin_age_pars_path: path to linAge_Paras.csv
    compute_bai_within_sample: if True, compute BAI using stats from the provided cohort.
                               For single-subject use-cases this will likely produce NaN.
    """
    data_mat = data_mat.reset_index(drop=True).copy()
    q_data_mat = q_data_mat.reset_index(drop=True).copy()

    # Derived quantities
    fs1_score = pop_pc_fi_fs1(q_data_mat)
    fs2_score = pop_pc_fi_fs2(q_data_mat)
    fs3_score = pop_pc_fi_fs3(q_data_mat)
    data_mat['fs1Score'] = fs1_score
    data_mat['fs2Score'] = fs2_score
    data_mat['fs3Score'] = fs3_score

    # Creatinine to albumin ratio and LDL
    data_mat['crAlbRat'] = pop_cr_alb_rat(data_mat)
    data_mat['LDLV'] = populate_ldl(data_mat, q_data_mat)

    # Digitize cotinine
    data_mat = digi_cot(data_mat)

    # Preserve original CRP for PhenoAge
    global original_crp
    original_crp = data_mat['LBXCRP'].copy()

    # Transformations
    data_mat['LBXCRP'] = np.log(data_mat['LBXCRP'])
    data_mat['SSBNP'] = np.log(data_mat['SSBNP'])

    # linAge model
    lin_age_pars = pd.read_csv(lin_age_pars_path)
    del_age = pop_lin_age(lin_age_pars, data_mat, q_data_mat)

    # Assemble outputs
    data_mat['chronAge'] = q_data_mat['RIDAGEYR']
    data_mat['delAge'] = del_age
    data_mat['linAge'] = data_mat['chronAge'] + data_mat['delAge']

    # PhenoAge and delta
    data_mat['phenoAge'] = calculate_pheno_age(data_mat, q_data_mat)
    data_mat['phenoDelAge'] = data_mat['phenoAge'] - data_mat['chronAge']

    # Biological Age Index (within-sample unless handled externally)
    if compute_bai_within_sample:
        data_mat = compute_bai(data_mat, q_data_mat, bin_width=bin_width)

    return data_mat

def main():
    # Main script
    q_data_file = "qDataMat.csv"
    data_file = "dataMat_test.csv"

    print("Reading qData matrix...")
    q_data_mat = pd.read_csv(q_data_file)
    print("Reading data matrix...")
    data_mat = pd.read_csv(data_file)

    keep_me = (q_data_mat['RIDAGEYR'] >= 40) & (q_data_mat['RIDAGEYR'] < 85)
    data_mat = data_mat[keep_me].reset_index(drop=True)
    q_data_mat = q_data_mat[keep_me].reset_index(drop=True)

    print("Calculating derived quantities...")
    fs1_score = pop_pc_fi_fs1(q_data_mat)
    fs2_score = pop_pc_fi_fs2(q_data_mat)
    fs3_score = pop_pc_fi_fs3(q_data_mat)
    data_mat['fs1Score'] = fs1_score
    data_mat['fs2Score'] = fs2_score
    data_mat['fs3Score'] = fs3_score

    print("Calculating creatinine to albumin ratio...")
    data_mat['crAlbRat'] = pop_cr_alb_rat(data_mat)

    print("Calculating LDL Cholesterol...")
    data_mat['LDLV'] = populate_ldl(data_mat, q_data_mat)

    print("Digitizing cotinine values...")
    data_mat = digi_cot(data_mat)

    # Save original CRP for PhenoAge calculation
    global original_crp
    original_crp = data_mat['LBXCRP'].copy()

    print("Applying parameter transformations...")
    data_mat['LBXCRP'] = np.log(data_mat['LBXCRP'])
    data_mat['SSBNP'] = np.log(data_mat['SSBNP'])

    print("Reading linAge model parameters...")
    lin_age_pars = pd.read_csv("linAge_Paras.csv")

    print("Running linAge model...")
    del_age = pop_lin_age(lin_age_pars, data_mat, q_data_mat)

    data_mat['chronAge'] = q_data_mat['RIDAGEYR']
    data_mat['delAge'] = del_age
    data_mat['linAge'] = data_mat['chronAge'] + data_mat['delAge']

    data_mat['phenoAge'] = calculate_pheno_age(data_mat, q_data_mat)
    data_mat['phenoDelAge'] = data_mat['phenoAge'] - data_mat['chronAge']

    # Compute BAI (Biological Age Index) using age/sex 10-year bins
    print("Computing Biological Age Index (BAI)...")
    data_mat = compute_bai(data_mat, q_data_mat, bin_width=10)

    print("Writing updated data matrix with PhenoAge...")
    data_mat.to_csv("dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv", index=False)

    # EDA and Plots
    print("Generating EDA plots...")

    # Set style
    sns.set(style="whitegrid")

    # Scatter plot: LinAge vs ChronAge
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_mat, x='chronAge', y='linAge', hue=q_data_mat['RIAGENDR'], palette='Set1')
    plt.title('LinAge vs Chronological Age')
    plt.savefig('linage_vs_chronage.png')
    plt.close()

    # Scatter plot: PhenoAge vs ChronAge
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_mat, x='chronAge', y='phenoAge', hue=q_data_mat['RIAGENDR'], palette='Set1')
    plt.title('PhenoAge vs Chronological Age')
    plt.savefig('phenoage_vs_chronage.png')
    plt.close()

    # Scatter plot: LinAge vs PhenoAge
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_mat, x='phenoAge', y='linAge', hue=q_data_mat['RIAGENDR'], palette='Set1')
    plt.title('LinAge vs PhenoAge')
    plt.savefig('linage_vs_phenoage.png')
    plt.close()

    # Correlation heatmap
    ages = data_mat[['chronAge', 'linAge', 'phenoAge', 'delAge', 'BAI']]
    corr = ages.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Ages')
    plt.savefig('age_correlations.png')
    plt.close()

    # Additional histogram for deltas
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_mat, x='delAge', kde=True, color='green', label='LinAge Delta')
    sns.histplot(data=data_mat, x='phenoDelAge', kde=True, color='red', label='PhenoAge Delta')
    plt.title('Distribution of Age Advancements (Deltas)')
    plt.legend()
    plt.savefig('delta_distributions.png')
    plt.close()

    # BAI distribution and category counts
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_mat, x='BAI', kde=True, color='purple')
    plt.title('Distribution of Biological Age Index (BAI)')
    plt.savefig('bai_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=data_mat, x='BAICategory', order=['Decelerated','Normal Aging','Accelerated','Highly Accelerated'])
    plt.title('BAI Risk Categories')
    plt.savefig('bai_categories.png')
    plt.close()

    # Decade-binned KDE distributions (ridgeline-style facets) by sex
    def plot_decade_kde(df, value_col, base_filename, xlabel):
        # Ensure categorical ordering for bins
        age_bins = [b for b in df['ageBin'].dropna().unique()]
        age_bins_order = sorted(age_bins, key=lambda s: int(s.split('-')[0]))

        for sex_label in ['Male', 'Female']:
            sub = df[df['sex'] == sex_label].copy()
            if sub.empty:
                continue
            g = sns.FacetGrid(sub, row='ageBin', hue='ageBin', row_order=age_bins_order,
                              aspect=4, height=1.2, palette='Spectral', sharex=True, sharey=False)
            g.map(sns.kdeplot, value_col, fill=True, alpha=0.9, linewidth=1)

            def _vline_mean(data, color, **kws):
                m = data[value_col].mean()
                plt.axvline(m, color='black', lw=1, ls='--')

            g.map_dataframe(_vline_mean)
            g.set_titles(row_template='{row_name}')
            g.set(xlabel=xlabel, ylabel='')
            for ax in g.axes.flatten():
                ax.set_yticks([])
            plt.subplots_adjust(hspace=0.05)
            g.fig.suptitle(f'{value_col} distributions by decade — {sex_label}', y=1.02)
            out_name = f"{base_filename}_{sex_label.lower()}.png"
            plt.savefig(out_name, bbox_inches='tight')
            plt.close()

    # Generate decade-binned distributions for LinAge and PhenoAge
    print('Generating decade-binned distributions by sex...')
    plot_decade_kde(data_mat, 'linAge', 'linage_decade_distributions', 'LinAge (years)')
    plot_decade_kde(data_mat, 'phenoAge', 'phenoage_decade_distributions', 'PhenoAge (years)')

    # Re-generate age_distributions.png with decade-binned ridgeline-style KDEs for chronAge, LinAge, PhenoAge
    print('Generating age_distributions.png with decade bins and Spectral colors...')
    def plot_combined_decade_kde(df):
        # Melt into long format
        long_df = pd.melt(df[['ageBin','sex','chronAge','linAge','phenoAge']].copy(),
                          id_vars=['ageBin','sex'],
                          value_vars=['chronAge','linAge','phenoAge'],
                          var_name='Measure', value_name='Age')
        # Order bins
        age_bins = sorted([b for b in df['ageBin'].dropna().unique()], key=lambda s: int(s.split('-')[0]))
        palette = sns.color_palette('Spectral', 3)
        color_map = {'chronAge': palette[0], 'linAge': palette[1], 'phenoAge': palette[2]}
        for sex_label in ['Male','Female']:
            sub = long_df[long_df['sex'] == sex_label].copy()
            g = sns.FacetGrid(sub, row='ageBin', row_order=age_bins, hue='Measure',
                              aspect=4, height=1.2, sharex=True, sharey=False, palette=color_map)
            g.map(sns.kdeplot, 'Age', fill=True, alpha=0.6, linewidth=1)
            g.add_legend(title='Measure')
            g.set_titles(row_template='{row_name}')
            g.set(xlabel='Age (years)', ylabel='')
            for ax in g.axes.flatten():
                ax.set_yticks([])
            plt.subplots_adjust(hspace=0.05)
            g.fig.suptitle(f'Age distributions by decade — {sex_label}', y=1.02)
            plt.savefig('age_distributions.png')
            plt.close()

    plot_combined_decade_kde(data_mat)

    print("Plots saved: age_distributions.png, linage_vs_chronage.png, phenoage_vs_chronage.png, linage_vs_phenoage.png, age_correlations.png, delta_distributions.png, bai_distribution.png, bai_categories.png, linage_decade_distributions_male.png, linage_decade_distributions_female.png, phenoage_decade_distributions_male.png, phenoage_decade_distributions_female.png")
    print("DONE!")

if __name__ == "__main__":
    main()
