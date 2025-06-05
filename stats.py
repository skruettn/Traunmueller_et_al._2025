import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests # For p-value correction

def plot_cellpose_data(bar_width=0.8, x_length=10, y_length=6):
    # Get all CSV files that match the pattern *_Cellpose.csv
    csv_files = glob.glob('DAPIexp.csv')

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Loop through each CSV file and append the data to all_data DataFrame
    for file in csv_files:
        df = pd.read_csv(file, dtype={'column_name': str}, low_memory=False)
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Check if 'Metadata_Region' column exists
    if 'Metadata_Region' not in all_data.columns:
        print(f"Error: The column 'Metadata_Region' was not found in your CSV file(s).")
        print(f"Available columns are: {all_data.columns.tolist()}")
        return  # Exit the function if the column is missing

    # Identify unique Metadata_Region values
    unique_regions = all_data['Metadata_Region'].unique()

    # Identify the column to plot
    intensity_columns = ['Intensity_MedianIntensity_RNAscope555', 'Intensity_MeanIntensity_RNAscope555', 'Intensity_MedianIntensity_RNAscope647', 'Intensity_MeanIntensity_RNAscope647', 'Intensity_MedianIntensity_wobg555', 'Intensity_MeanIntensity_wobg555', 'Intensity_MedianIntensity_wobg647', 'Intensity_MeanIntensity_wobg647']


    # Get a list of unique Metadata_TP values for consistent ordering
    unique_tps = sorted(all_data['Metadata_TP'].unique())

    # define the order of TP plotting
    TP_order = ['HC', '1h', '8h', 'c6h']

    # Plot for each unique Metadata_Region
    for region in unique_regions:
        region_data = all_data[all_data['Metadata_Region'] == region].copy()

        for intensity_column in intensity_columns: # Corrected variable name here

            # Calculate mean intensity per Metadata_animal and Metadata_TP
            mean_per_animal = region_data.groupby(['Metadata_TP', 'Metadata_animal'])[intensity_column].mean().reset_index()

            # --- Violin Plot ---
            plt.figure(figsize=(x_length, y_length))
            sns.violinplot(x='Metadata_TP', y=intensity_column, data=region_data, palette='viridis', order=TP_order)
            sns.stripplot(x='Metadata_TP', y=intensity_column, data=region_data, color='black', alpha=0.3, order=TP_order) # Added stripplot for individual data points
            sns.stripplot(x='Metadata_TP', y=intensity_column, data=mean_per_animal, color='red', marker='o', size=8, edgecolor='black', linewidth=1, order=TP_order)
            plt.title(f'{intensity_column} (Violin Plot) in {region}')
            plt.xlabel('Metadata_TP')
            plt.ylabel(intensity_column)
            plt.ylim(-0.2, 1.2)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{intensity_column}_ViolinPlot_Region_{region}.pdf')
            plt.close()


# Example usage with custom bar width and axis lengths
plot_cellpose_data(bar_width=0.2, x_length=8, y_length=6)

def _get_significance_stars(p_value):
    """Returns significance stars based on p-value thresholds."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns' # not significant

def generate_summary_csv(input_csv_file='Image.csv', output_csv_file='summary.csv'):
    """
    Generates a summary CSV file containing:
    1. Detailed summary grouped by Metadata_animal, Metadata_Region, Metadata_TP.
    2. 'Totals' summary pooling Metadata_animal data, grouped by Metadata_Region, Metadata_TP.
    3. Kruskal-Wallis H-test results for overall comparison.
    4. Pairwise Mann-Whitney U test results with Bonferroni correction and significance stars.

    Args:
        input_csv_file (str): The name of the input CSV file (e.g., 'DAPIexp.csv').
        output_csv_file (str): The name of the output summary CSV file.
    """
    try:
        df = pd.read_csv(input_csv_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_file}' was not found.")
        return

    # Define columns for aggregation
    sum_columns = ['Count_DAPIexp', 'Count_DAPICP']
    mean_columns = [
        'Mean_DAPIexp_Intensity_MedianIntensity_RNAscope555',
        'Mean_DAPIexp_Intensity_MedianIntensity_RNAscope647',
        'Mean_DAPIexp_Intensity_MedianIntensity_wobg555',
        'Mean_DAPIexp_Intensity_MedianIntensity_wobg647'
    ]
    # Define intensity columns for statistical tests (using the full names as they appear in the CSV)
    intensity_cols_for_stats = [
        'Mean_DAPIexp_Intensity_MedianIntensity_RNAscope555',
        'Mean_DAPIexp_Intensity_MedianIntensity_RNAscope647',
        'Mean_DAPIexp_Intensity_MedianIntensity_wobg555',
        'Mean_DAPIexp_Intensity_MedianIntensity_wobg647'
    ]
    TP_order = ['HC', '1h', '8h', 'c6h'] # Consistent TP order

    # Create aggregation dictionary
    agg_dict = {col: 'sum' for col in sum_columns}
    agg_dict.update({col: 'mean' for col in mean_columns})

    # --- Generate the first summary table (detailed) ---
    group_by_columns_detailed = ['Metadata_animal', 'Metadata_Region', 'Metadata_TP']
    required_columns_detailed = group_by_columns_detailed + sum_columns + mean_columns
    missing_columns_detailed = [col for col in required_columns_detailed if col not in df.columns]
    if missing_columns_detailed:
        print(f"Error: Missing columns for detailed summary: {missing_columns_detailed}. Available: {df.columns.tolist()}")
        return
    summary_df = df.groupby(group_by_columns_detailed).agg(agg_dict).reset_index()

    # --- Generate the second summary table (totals, pooling animal data) ---
    group_by_columns_totals = ['Metadata_Region', 'Metadata_TP']
    required_columns_totals = group_by_columns_totals + sum_columns + mean_columns
    missing_columns_totals = [col for col in required_columns_totals if col not in df.columns]
    if missing_columns_totals:
        print(f"Error: Missing columns for totals summary: {missing_columns_totals}. Available: {df.columns.tolist()}")
        return
    totals_df = df.groupby(group_by_columns_totals).agg(agg_dict).reset_index()
    totals_df.insert(0, 'Metadata_animal', 'Pooled_Total')

    # --- Perform Statistical Calculations ---
    kruskal_results = []
    pairwise_results = []
    unique_regions = df['Metadata_Region'].unique() # Get all regions from the full dataframe

    for region in unique_regions:
        region_data = df[df['Metadata_Region'] == region].copy()
        
        for intensity_col in intensity_cols_for_stats:
            # Check if the intensity column exists and has non-NA data
            if intensity_col not in region_data.columns or region_data[intensity_col].isnull().all():
                print(f"Warning: Skipping stats for {region}, {intensity_col} due to missing or all NaN data.")
                continue

            # Prepare data for Kruskal-Wallis
            data_for_kruskal = [region_data[region_data['Metadata_TP'] == tp][intensity_col].dropna().tolist()
                                for tp in TP_order if not region_data[region_data['Metadata_TP'] == tp][intensity_col].dropna().empty]

            if len(data_for_kruskal) >= 2: # Kruskal-Wallis needs at least two groups
                try:
                    h_statistic, p_kruskal = stats.kruskal(*data_for_kruskal)
                    kruskal_results.append({
                        'Region': region,
                        'Intensity_Column': intensity_col,
                        'Kruskal_Wallis_H_Statistic': h_statistic,
                        'Kruskal_Wallis_P_Value': p_kruskal,
                        'Overall_Significance': _get_significance_stars(p_kruskal)
                    })
                except ValueError as e:
                    print(f"Could not perform Kruskal-Wallis for {region}, {intensity_col}: {e}")
            else:
                print(f"Skipping Kruskal-Wallis for {region}, {intensity_col}: Not enough groups with data.")

            # Prepare data for Pairwise Mann-Whitney U
            p_values_for_correction = []
            comparisons = []

            for tp1, tp2 in combinations(TP_order, 2):
                group1_data = region_data[region_data['Metadata_TP'] == tp1][intensity_col].dropna().tolist()
                group2_data = region_data[region_data['Metadata_TP'] == tp2][intensity_col].dropna().tolist()

                if len(group1_data) > 0 and len(group2_data) > 0:
                    try:
                        u_statistic, p_mw = stats.mannwhitneyu(group1_data, group2_data)
                        p_values_for_correction.append(p_mw)
                        comparisons.append({'tp1': tp1, 'tp2': tp2, 'u_stat': u_statistic})
                    except ValueError as e:
                        # Happens if one group has constant data (e.g., all same value), Mann-Whitney doesn't work
                        p_values_for_correction.append(1.0) # Assign 1.0 to ensure it's not significant if test failed
                        comparisons.append({'tp1': tp1, 'tp2': tp2, 'u_stat': float('nan')})
                        print(f"Could not perform Mann-Whitney U for {region}, {intensity_col}, {tp1} vs {tp2}: {e}")
                else:
                    p_values_for_correction.append(1.0) # No data for comparison, not significant
                    comparisons.append({'tp1': tp1, 'tp2': tp2, 'u_stat': float('nan')})
                    print(f"Skipping Mann-Whitney U for {region}, {intensity_col}, {tp1} vs {tp2}: Insufficient data.")

            if p_values_for_correction:
                # Apply Bonferroni correction
                reject, corrected_p_values, _, _ = multipletests(p_values_for_correction, alpha=0.05, method='bonferroni')

                for i, comp_info in enumerate(comparisons):
                    corrected_p = corrected_p_values[i]
                    pairwise_results.append({
                        'Region': region,
                        'Intensity_Column': intensity_col,
                        'Timepoint_1': comp_info['tp1'],
                        'Timepoint_2': comp_info['tp2'],
                        'Mann_Whitney_U_Statistic': comp_info['u_stat'],
                        'P_Value_Uncorrected': p_values_for_correction[i],
                        'P_Value_Corrected_Bonferroni': corrected_p,
                        'Significance_Stars': _get_significance_stars(corrected_p)
                    })

    kruskal_results_df = pd.DataFrame(kruskal_results)
    pairwise_results_df = pd.DataFrame(pairwise_results)

    # --- Write all tables to the same CSV file ---
    with open(output_csv_file, 'w', newline='') as f:
        f.write("Detailed Summary (Grouped by Animal, Region, TP):\n")
        summary_df.to_csv(f, index=False, header=True)
        f.write("\n\n") # Add some blank lines for separation

        f.write("Totals Summary (Pooled Animals, Grouped by Region, TP):\n")
        totals_df.to_csv(f, index=False, header=True)
        f.write("\n\n") # Add some blank lines for separation

        if not kruskal_results_df.empty:
            f.write("Overall Kruskal-Wallis H-Test Results:\n")
            kruskal_results_df.to_csv(f, index=False, header=True)
            f.write("\n\n") # Add some blank lines for separation
        else:
            f.write("No Kruskal-Wallis results available.\n\n")

        if not pairwise_results_df.empty:
            f.write("Pairwise Mann-Whitney U Test Results (Bonferroni Corrected):\n")
            pairwise_results_df.to_csv(f, index=False, header=True)
            f.write("\n\n") # Add some blank lines for separation
        else:
            f.write("No Pairwise Mann-Whitney U results available.\n\n")


    print(f"Summary data with 'Detailed', 'Totals', and Statistical tables saved to '{output_csv_file}' successfully.")

# Example usage with custom bar width and axis lengths
plot_cellpose_data(bar_width=0.2, x_length=8, y_length=6)

# Generate the summary CSV with two tables and statistics
generate_summary_csv(input_csv_file='Image.csv', output_csv_file='summary.csv')