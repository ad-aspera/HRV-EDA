The following folder concerns the results of secondary analysis study conducted for Yr3 Group Project for fulfilment of Molecular Bioengineering MEng Degree at Imperial College London.

The report itself can be found in parent folder.

The dataset analysed was  [Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus  2023](https://data.mendeley.com/datasets/9c47vwvtss/4).

Accompanying ECG RR detection algorithm can be found in sister folder.

Data file can be found separately.

## Disclaimer

Note that an error was likely introduced before or after the refactoring process of the notebooks.
Consequently, the results of statistical significance testing have changed and expanded to include LF Power as a statistically different metric. Due refactoring happening past the submission deadline, this change is not reflected in the final report submitted. 

Given that the error resulted in a false negative rather than a false positive result and the small dataset size, it does not subtract from the final conclusion that the role of HRV in DPN identification should be investigated further.


## Key Notebooks

Here is a list of key notebooks used during the data analysis stage of the project

* [cheng_to_single_table.ipynb](cheng_to_single_table.ipynb) - Converts metric readings to a easier to access format.
* [Demographics2.ipynb](Demographics2.ipynb) - analysis of patient clinical markers.
* [RR_to_metrics.ipynb](RR_to_metrics.ipynb) - Derives the standard metrics from the RR peaks.
* [median_approach.ipynb](median_approach.ipynb) - derives median values for each patients metric distributions and investigates them for statistical difference.
* [LDA.ipynb](LDA.ipynb) - performs LDA to investigate separation between groups.

Here are the less important notebooks:
* [report_images_generation.ipynb](generation of images used for the report)
* [decile_approach.ipynb](decile_approach.ipynb) - investigation on whether metric deciles can provide more robust channels. Abandoned due to small sample size.
* [SVM_median.ipynb](SVM_median.ipynb) - Attempts to build a more sophisticated classificatory. Abandoned due to small sample size causing overhitting.
* [Bayesian_approach.ipynb](Bayesian_approach.ipynb) - Attempts to predict whether using random metric from individual would still result in statistical significance. The high within-patient metric variance and small sample size, has resulted in false positives. However, the idea is worth investigating with larger patient size for diagnostic purposes.
