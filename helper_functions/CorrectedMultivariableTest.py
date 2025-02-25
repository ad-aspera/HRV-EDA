import scipy.stats as stats 
import numpy as np
import pandas as pd


class CorrectedMultivariableTest:
    def __init__(self, data:pd.DataFrame, group_col:str, value_col:str, cat_col:str, cat_val_1:str, cat_val_2:str, alpha=0.05):
        """
        group col - defines separate groups for each of which the test will be performed
        value col - the column containing the target variable
        cat col - the column containing the categories to be compared
        cat val 1 - the first category to be compared
        cat val 2 - the second category to be compared
        alpha - the significance level for the Benjamini-Hochberg correction
        """
        self.data = data
        self.group_col = group_col
        self.target_col = value_col
        self.cat_col = cat_col
        self.cat1 = cat_val_1
        self.cat2 = cat_val_2
        self.alpha = alpha

    def perform_t_tests(self):
        """
        Perform t-tests between two categories within each group and apply Benjamini-Hochberg correction.
        """
        test_results = self._t_tests()
        test_results_df = self._apply_bh_correction(test_results)
        return test_results_df
    
    def perform_ManWhitney_U_tests(self):
        """
        Perform Mann-Whitney U tests between two categories within each group and apply Benjamini-Hochberg correction.
        """
        test_results = self._mann_whitney_u_tests()
        test_results_df = self._apply_bh_correction(test_results)
        return test_results_df
    
    def _mann_whitney_u_tests(self):
        """
        Perform Mann-Whitney U tests between two categories within each group.
        """
        group_ids = self.data[self.group_col].unique()
        test_results = []

        for group_id in group_ids:
            group_data = self.data[self.data[self.group_col] == group_id]
            cat1_data = group_data[group_data[self.cat_col] == self.cat1][self.target_col]
            cat2_data = group_data[group_data[self.cat_col] == self.cat2][self.target_col]

            stat, p_val = stats.mannwhitneyu(cat1_data, cat2_data)

            test_results.append({
                self.group_col: group_id,
                'U_statistic': stat,
                'p_value': p_val,
                'significant': p_val < self.alpha,
            })

        return test_results

    def _t_tests(self):
        """
        Perform t-tests between two categories within each group.
        """
        group_ids = self.data[self.group_col].unique()
        test_results = []

        for group_id in group_ids:
            group_data = self.data[self.data[self.group_col] == group_id]
            cat1_data = group_data[group_data[self.cat_col] == self.cat1][self.target_col]
            cat2_data = group_data[group_data[self.cat_col] == self.cat2][self.target_col]

            stat, p_val = stats.ttest_ind(cat1_data, cat2_data, equal_var=False)

            test_results.append({
                self.group_col: group_id,
                't_statistic': stat,
                'p_value': p_val,
                'significant': p_val < self.alpha,
        
            })

        return test_results

    def _apply_bh_correction(self, test_results:pd.DataFrame)->pd.DataFrame:
        """
        Apply Benjamini-Hochberg correction to a dataframe p-values.
        Return the result with two extra column - BH_corrected_p_value and BH_Significant.
        """
        p_values = [result['p_value'] for result in test_results]
        p_values_sorted_indices = np.argsort(p_values)
        n = len(p_values)

        for i, index in enumerate(p_values_sorted_indices):
            rank = i + 1
            bh_threshold = (rank / n) * self.alpha
            test_results[index]['BH_corrected_p_value'] = p_values[index] * self.alpha / bh_threshold
            test_results[index]['BH_Significant'] = test_results[index]['BH_corrected_p_value'] < self.alpha

        test_results_df = pd.DataFrame(test_results)
        test_results_df.set_index(self.group_col, inplace=True)
        test_results_df.sort_index(inplace=True)

        return test_results_df
    

if __name__ == "__main__":
    # Simulate some data
    np.random.seed(42)
    group_ids = ['Group1', 'Group2', 'Group3']
    categories = ['Cat1', 'Cat2']
    data = []

    for group in group_ids:
        for cat in categories:
            for _ in range(30):  # 30 samples per category per group
                data.append({
                    'Group': group,
                    'Category': cat,
                    'Value': np.random.normal(loc=0 if cat == 'Cat1' else 1, scale=1)
                })

    df = pd.DataFrame(data)

    # Initialize the CorrectedMultivariableTest class
    test = CorrectedMultivariableTest(data=df, group_col='Group', value_col='Value', cat_col='Category', cat_val_1='Cat1', cat_val_2='Cat2')

    # Perform t-tests
    t_test_results = test.perform_t_tests()
    print("T-test results with Benjamini-Hochberg correction:")
    print(t_test_results)

    # Perform Mann-Whitney U tests
    mw_test_results = test.perform_ManWhitney_U_tests()
    print("\nMann-Whitney U test results with Benjamini-Hochberg correction:")
    print(mw_test_results)