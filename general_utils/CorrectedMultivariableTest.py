import scipy.stats as stats 
import numpy as np
import pandas as pd
import math
from typing import Callable, List, Dict, Any, Tuple

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

    def perform_test(self, test_method: str, **kwargs):
        """
        Generic method to perform statistical tests and apply correction
        """
        test_methods = {
            't_test': self._t_tests,
            'mann_whitney': self._mann_whitney_u_tests,
            'permutation': lambda: self._permutation_test(**kwargs)
        }
        
        if test_method not in test_methods:
            raise ValueError(f"Test method '{test_method}' not supported")
            
        test_results = test_methods[test_method]()
        return self._apply_bh_correction(test_results)
    
    def _get_group_category_data(self, group_id):
        """Extract data for a specific group and both categories"""
        group_data = self.data[self.data[self.group_col] == group_id]
        cat1_data = group_data[group_data[self.cat_col] == self.cat1][self.target_col]
        cat2_data = group_data[group_data[self.cat_col] == self.cat2][self.target_col]
        return cat1_data, cat2_data
    
    def _run_test_for_groups(self, test_func: Callable) -> List[Dict[str, Any]]:
        """Run a test function for each group and collect results"""
        group_ids = self.data[self.group_col].unique()
        test_results = []

        for group_id in group_ids:
            cat1_data, cat2_data = self._get_group_category_data(group_id)
            result = test_func(cat1_data, cat2_data)
            result[self.group_col] = group_id
            test_results.append(result)

        return test_results

    def _mann_whitney_u_tests(self):
        """Perform Mann-Whitney U tests between two categories within each group."""
        def run_test(cat1_data, cat2_data):
            stat, p_val = stats.mannwhitneyu(cat1_data, cat2_data)
            return {
                'U_statistic': stat,
                'p_value': p_val,
                'significant': p_val < self.alpha,
            }
        
        return self._run_test_for_groups(run_test)

    def _t_tests(self):
        """Perform t-tests between two categories within each group."""
        def run_test(cat1_data, cat2_data):
            stat, p_val = stats.ttest_ind(cat1_data, cat2_data, equal_var=False)
            return {
                't_statistic': stat,
                'p_value': p_val,
                'significant': p_val < self.alpha,
            }
        
        return self._run_test_for_groups(run_test)
    
    def _permutation_test(self, statistic, n_permutations=1000):
        """Perform permutation test between two categories within each group."""
        def run_test(cat1_data, cat2_data):
            result = stats.permutation_test(
                (cat1_data, cat2_data), 
                statistic=statistic, 
                permutation_type='independent', 
                n_resamples=n_permutations
            )
            return {
                'distribution': result.null_distribution,
                'perm_statistic': result.statistic,
                'p_value': result.pvalue,
                'significant': result.pvalue < self.alpha,
            }
        
        return self._run_test_for_groups(run_test)

    def _apply_bh_correction(self, test_results):
        """
        Apply Benjamini-Hochberg correction to a data frame p-values.
        Return the result with extra columns for BH correction.
        """
        if not isinstance(test_results, pd.DataFrame):
            test_results = pd.DataFrame(test_results)

        n = len(test_results)
        test_results['rank'] = test_results['p_value'].rank(ascending=False, method='first')
        test_results['BH_threshold'] = (test_results['rank'] / n) * self.alpha
        test_results['BH_Significant'] = test_results['p_value'] < test_results['BH_threshold']

        test_results.set_index(self.group_col, inplace=True)
        return test_results.sort_index()

    # Maintain backward compatibility
    def perform_t_tests(self):
        return self.perform_test('t_test')
    
    def perform_ManWhitney_U_tests(self):
        return self.perform_test('mann_whitney')
    
    def permutation_test(self, statistic, n_permutations=1000):
            
        return self.perform_test('permutation', statistic=statistic, n_permutations=n_permutations)
