from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt


def get_vif(data:pd.DataFrame, exclude_columns:list[str] = None) -> pd.DataFrame:
    vif_data = pd.DataFrame()

    # Store original columns
    original_columns = data.columns.tolist()
    
    vif_data["feature"] = original_columns
    vif_data["Full VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(original_columns))]
    
    if exclude_columns is not None:
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]
            
        # Create a filtered dataset without excluded columns
        filtered_data = data.drop(exclude_columns, axis=1)
        filtered_columns = filtered_data.columns.tolist()
        
        # Create mapping between original and filtered column indices
        filtered_vif_values = {}
        for i, col in enumerate(filtered_columns):
            filtered_vif_values[col] = variance_inflation_factor(filtered_data.values, i)
        
        # Build the Excluded VIF column, with NaN for excluded columns
        excluded_vif = []
        for col in original_columns:
            if col in filtered_columns:
                excluded_vif.append(filtered_vif_values[col])
            else:
                excluded_vif.append(np.nan)
                
        vif_data["Excluded VIF"] = excluded_vif
    # Sort by Excluded VIF if it exists, otherwise by Full VIF
    if "Excluded VIF" in vif_data.columns:
        vif_data = vif_data.sort_values(by="Excluded VIF", ascending=False, na_position='last')
    else:
        vif_data = vif_data.sort_values(by="Full VIF", ascending=False)
    return vif_data
    

def plot_vif(data:pd.DataFrame, height=5):
    nan_columns = data[data.isna().any(axis=1)]["feature"].tolist()


    value_vars = ["Full VIF"] if "Excluded VIF" not in data.columns else ["Full VIF", "Excluded VIF"]
    melted_data = data.melt(id_vars=["feature"], value_vars=value_vars, var_name="VIF Type", value_name="VIF Value")
    plt.figure(figsize=(8, height))
   
    sns.barplot(x="VIF Value", y="feature", hue="VIF Type", data=melted_data, dodge=True)
    plt.title(f"VIF for wine variables; Excluded VIF for {nan_columns}")
    plt.legend()
    plt.xscale('log')
    plt.grid(axis='x', linestyle='-', alpha=0.5)
    plt.grid(axis='x', linestyle='--', which='minor',alpha = 0.5)
    plt.tight_layout()
    plt.show()



def lda_vif_exclude(data, exclude):
    data = data.copy()
    if isinstance(exclude, str):
        exclude = [exclude]

    plot_vif(get_vif(data.drop(columns =['id', 'DPN']), exclude_columns=exclude))
    data = data.drop(columns=exclude)
    indicator_columns = data.columns[2:]
    lda_df, lda = perform_lda(data, indicator_columns)
    plot_linear_lda(lda_df, f"HRV LDA with excluded {exclude}")
    show_linear_lda_stats(lda, indicator_columns)




def perform_lda(data, columns = None, n_components=1):
    target = data["DPN"]
    if columns is not None:
        data = data[columns]

    data = 2 * (data - data.min()) / (data.max() - data.min()) - 1

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_data = lda.fit_transform(data, target)
    lda_df = pd.DataFrame(lda_data)
    lda_df['Cluster'] = target

   # print(dir(lda_data))
    return lda_df, lda


def plot_linear_lda(lda_df, title = 'LDA Results', cluster_colors = ['blue', 'red']):
    plt.figure(figsize=(10, 1))
    sns.scatterplot(x=lda_df[0], y=[0] * len(lda_df), hue=lda_df['Cluster'], palette=cluster_colors, s=100, alpha=0.7)
    plt.yticks([])
    plt.xlabel('LDA Component 1')
    plt.title(title)
    
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.show()

def show_linear_lda_stats(lda, columns):
    #print("Explained variance ratio: ", lda.explained_variance_ratio_)
    #print("Intercept: ", lda.intercept_)
    #print("Priors", lda.priors_)

    df = pd.DataFrame({
        'Columns': columns,
        'LDA Coef': lda.coef_[0],
        'Means': lda.means_[0],
    })
    # Sort the DataFrame by coefficient values for better visualization
    df_sorted = df.sort_values(by='LDA Coef', ascending=False)

    # Plot the coefficients and means as horizontal bars
    plt.figure(figsize=(12, 4))

    # First subplot for LDA coefficients
    plt.subplot(1, 2, 1)
    plt.barh(df_sorted['Columns'], df_sorted['LDA Coef'])
    plt.title('LDA Coefficients')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Second subplot for means
    plt.subplot(1, 2, 2)
    plt.barh(df_sorted['Columns'], df_sorted['Means'])
    plt.title('LDA Means')
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
   # display(df)
    return df