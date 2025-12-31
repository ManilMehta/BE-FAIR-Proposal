import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Data from Table 2 of the BE-FAIR paper
# Creating mock dataset based on published statistics

def create_mock_dataset():
    """
    Create a mock dataset based on Table 2 statistics from the paper.
    This simulates the actual patient data for analysis.
    """
    
    # Race/Ethnicity groups with sample sizes and event rates
    race_data = {
        'White/Caucasian': {'n': 68061, 'ed_rate': 0.020, 'hosp_rate': 0.044},
        'AIAN': {'n': 415, 'ed_rate': 0.019, 'hosp_rate': 0.060},
        'Black': {'n': 6207, 'ed_rate': 0.052, 'hosp_rate': 0.089},
        'Multi-racial': {'n': 4180, 'ed_rate': 0.023, 'hosp_rate': 0.044},
        'AAPI': {'n': 15172, 'ed_rate': 0.016, 'hosp_rate': 0.036},
        'Hispanic': {'n': 15806, 'ed_rate': 0.025, 'hosp_rate': 0.049},
        'Other': {'n': 4470, 'ed_rate': 0.022, 'hosp_rate': 0.046}
    }
    
    # HPI (Healthy Places Index) quartiles
    hpi_data = {
        'HPI 0-25%': {'n': 28675, 'ed_rate': 0.037, 'hosp_rate': 0.069},
        'HPI 25-50%': {'n': 28677, 'ed_rate': 0.021, 'hosp_rate': 0.047},
        'HPI 50-75%': {'n': 28541, 'ed_rate': 0.017, 'hosp_rate': 0.038},
        'HPI 75-100%': {'n': 28418, 'ed_rate': 0.014, 'hosp_rate': 0.030}
    }
    
    return race_data, hpi_data

def calculate_predicted_values(race_data, hpi_data):
    """
    Calculate predicted vs actual values based on calibration data from Figure 1 & 2.
    Using the log odds ratios reported in the paper.
    """
    
    # Predicted probabilities from the paper's calibration analysis
    # These values are extracted from the calibration intercepts
    race_predictions = {
        'White/Caucasian': {'ed_pred': 0.020, 'hosp_pred': 0.044},  # Reference group
        'AIAN': {'ed_pred': 0.022, 'hosp_pred': 0.055},
        'Black': {'ed_pred': 0.045, 'hosp_pred': 0.129},  # Significantly different
        'Multi-racial': {'ed_pred': 0.018, 'hosp_pred': 0.245},  # Significantly different
        'AAPI': {'ed_pred': 0.012, 'hosp_pred': 0.120},  # Significantly different
        'Hispanic': {'ed_pred': 0.020, 'hosp_pred': 0.133},  # Significantly different
        'Other': {'ed_pred': 0.019, 'hosp_pred': 0.041}
    }
    
    hpi_predictions = {
        'HPI 0-25%': {'ed_pred': 0.030, 'hosp_pred': 0.178},  # Significantly different
        'HPI 25-50%': {'ed_pred': 0.019, 'hosp_pred': 0.129},  # Significantly different
        'HPI 50-75%': {'ed_pred': 0.016, 'hosp_pred': 0.045},
        'HPI 75-100%': {'ed_pred': 0.013, 'hosp_pred': 0.035}
    }
    
    return race_predictions, hpi_predictions

def compute_fairness_metrics(actual_rate, predicted_rate, group_name, outcome_type):
    """
    Compute key fairness metrics for a subgroup.
    """
    
    # Prediction error (bias)
    prediction_error = predicted_rate - actual_rate
    
    # Relative error (percentage)
    if actual_rate > 0:
        relative_error = (prediction_error / actual_rate) * 100
    else:
        relative_error = 0
    
    # Categorize bias direction
    if abs(relative_error) < 10:
        bias_category = "Well-calibrated"
    elif prediction_error > 0:
        bias_category = "Over-prediction"
    else:
        bias_category = "Under-prediction"
    
    return {
        'Group': group_name,
        'Outcome': outcome_type,
        'Actual_Rate': actual_rate,
        'Predicted_Rate': predicted_rate,
        'Absolute_Error': abs(prediction_error),
        'Relative_Error_%': relative_error,
        'Bias_Direction': bias_category
    }

def analyze_subgroup_fairness():
    """
    Main analysis function: compute fairness metrics across all subgroups.
    """
    
    print("=" * 80)
    print("BE-FAIR SUBGROUP FAIRNESS ANALYSIS")
    print("Preliminary Analysis of Prediction Disparities")
    print("=" * 80)
    print()
    
    # Load data
    race_data, hpi_data = create_mock_dataset()
    race_predictions, hpi_predictions = calculate_predicted_values(race_data, hpi_data)
    
    # Storage for results
    all_metrics = []
    
    # Analyze Race/Ethnicity subgroups
    print("ANALYSIS 1: Race/Ethnicity Disparities")
    print("-" * 80)
    
    for race, stats in race_data.items():
        preds = race_predictions[race]
        
        # ED visits
        ed_metrics = compute_fairness_metrics(
            stats['ed_rate'], preds['ed_pred'], race, 'ED Visit'
        )
        all_metrics.append(ed_metrics)
        
        # Hospitalizations
        hosp_metrics = compute_fairness_metrics(
            stats['hosp_rate'], preds['hosp_pred'], race, 'Hospitalization'
        )
        all_metrics.append(hosp_metrics)
    
    # Analyze HPI subgroups
    print("\nANALYSIS 2: Social Vulnerability (HPI) Disparities")
    print("-" * 80)
    
    for hpi, stats in hpi_data.items():
        preds = hpi_predictions[hpi]
        
        # ED visits
        ed_metrics = compute_fairness_metrics(
            stats['ed_rate'], preds['ed_pred'], hpi, 'ED Visit'
        )
        all_metrics.append(ed_metrics)
        
        # Hospitalizations
        hosp_metrics = compute_fairness_metrics(
            stats['hosp_rate'], preds['hosp_pred'], hpi, 'Hospitalization'
        )
        all_metrics.append(hosp_metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    return results_df

def visualize_disparities(results_df):
    """
    Create visualizations of subgroup disparities.
    """
    
    # Split by analysis type
    race_results = results_df[~results_df['Group'].str.contains('HPI')]
    hpi_results = results_df[results_df['Group'].str.contains('HPI')]
    
    # Figure 1: Race/Ethnicity Disparities
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: ED Visit Errors by Race
    ed_race = race_results[race_results['Outcome'] == 'ED Visit']
    axes[0, 0].barh(ed_race['Group'], ed_race['Relative_Error_%'], 
                     color=['red' if x < -10 else 'green' if x > 10 else 'gray' 
                            for x in ed_race['Relative_Error_%']])
    axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Relative Prediction Error (%)')
    axes[0, 0].set_title('ED Visit Prediction Errors by Race/Ethnicity')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Subplot 2: Hospitalization Errors by Race
    hosp_race = race_results[race_results['Outcome'] == 'Hospitalization']
    axes[0, 1].barh(hosp_race['Group'], hosp_race['Relative_Error_%'],
                     color=['red' if x < -10 else 'green' if x > 10 else 'gray' 
                            for x in hosp_race['Relative_Error_%']])
    axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Relative Prediction Error (%)')
    axes[0, 1].set_title('Hospitalization Prediction Errors by Race/Ethnicity')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Subplot 3: ED Visit Errors by HPI
    ed_hpi = hpi_results[hpi_results['Outcome'] == 'ED Visit']
    axes[1, 0].barh(ed_hpi['Group'], ed_hpi['Relative_Error_%'],
                     color=['red' if x < -10 else 'green' if x > 10 else 'gray' 
                            for x in ed_hpi['Relative_Error_%']])
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Relative Prediction Error (%)')
    axes[1, 0].set_title('ED Visit Prediction Errors by Social Vulnerability (HPI)')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Subplot 4: Hospitalization Errors by HPI
    hosp_hpi = hpi_results[hpi_results['Outcome'] == 'Hospitalization']
    axes[1, 1].barh(hosp_hpi['Group'], hosp_hpi['Relative_Error_%'],
                     color=['red' if x < -10 else 'green' if x > 10 else 'gray' 
                            for x in hosp_hpi['Relative_Error_%']])
    axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Relative Prediction Error (%)')
    axes[1, 1].set_title('Hospitalization Prediction Errors by Social Vulnerability (HPI)')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('subgroup_fairness_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'subgroup_fairness_analysis.png'")
    
    return fig

def generate_summary_table(results_df):
    """
    Generate a publication-ready summary table.
    """
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Prediction Errors by Subgroup")
    print("=" * 80)
    
    # Focus on hospitalization (higher impact outcome)
    hosp_results = results_df[results_df['Outcome'] == 'Hospitalization'].copy()
    
    # Sort by absolute error
    hosp_results = hosp_results.sort_values('Absolute_Error', ascending=False)
    
    # Format for display
    summary = hosp_results[['Group', 'Actual_Rate', 'Predicted_Rate', 
                            'Absolute_Error', 'Relative_Error_%', 'Bias_Direction']]
    summary['Actual_Rate'] = summary['Actual_Rate'].apply(lambda x: f"{x:.3f}")
    summary['Predicted_Rate'] = summary['Predicted_Rate'].apply(lambda x: f"{x:.3f}")
    summary['Absolute_Error'] = summary['Absolute_Error'].apply(lambda x: f"{x:.3f}")
    summary['Relative_Error_%'] = summary['Relative_Error_%'].apply(lambda x: f"{x:.1f}%")
    
    print(summary.to_string(index=False))
    print()
    
    return summary

def main():
    """
    Execute complete analysis pipeline.
    """
    
    # Run analysis
    results_df = analyze_subgroup_fairness()
    
    # Generate visualizations
    visualize_disparities(results_df)
    
    # Generate summary table
    summary = generate_summary_table(results_df)
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    hosp_results = results_df[results_df['Outcome'] == 'Hospitalization']
    
    # Find groups with largest disparities
    worst_underprediction = hosp_results.loc[hosp_results['Relative_Error_%'].idxmin()]
    worst_overprediction = hosp_results.loc[hosp_results['Relative_Error_%'].idxmax()]
    
    print(f"\n1. LARGEST UNDER-PREDICTION:")
    print(f"   Group: {worst_underprediction['Group']}")
    print(f"   Model predicted {worst_underprediction['Predicted_Rate']:.1%} but actual was {worst_underprediction['Actual_Rate']:.1%}")
    print(f"   Relative error: {worst_underprediction['Relative_Error_%']:.1f}%")
    
    print(f"\n2. LARGEST OVER-PREDICTION:")
    print(f"   Group: {worst_overprediction['Group']}")
    print(f"   Model predicted {worst_overprediction['Predicted_Rate']:.1%} but actual was {worst_overprediction['Actual_Rate']:.1%}")
    print(f"   Relative error: {worst_overprediction['Relative_Error_%']:.1f}%")
    
    # Count groups with significant bias
    significant_bias = hosp_results[abs(hosp_results['Relative_Error_%']) > 10]
    print(f"\n3. GROUPS WITH SIGNIFICANT BIAS (>10% error): {len(significant_bias)} out of {len(hosp_results)}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Save results
    results_df.to_csv('fairness_metrics.csv', index=False)
    print("\n✓ Full results saved to 'fairness_metrics.csv'")
    
    return results_df

if __name__ == "__main__":
    results = main()