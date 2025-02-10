# Key files

* [RR_to_metrics.ipynb](RR_to_metrics.ipynb) is used to calculate metrics from an array of RR values.


* [eda.ipynb](eda.ipynb) is the main analysis of the signal.

* [html_plots/](html_plots) stores the generated images.


# HRV metrics from article derived peaks

See [article_derived_HRV_data](article_derived_HRV_data) for the data

# Signal as pd Series
Most of functions in this repo treat signal as a Pandas Series:
* Values are RR intervals in milliseconds
* Index is time domain in milliseconds

These formats can very easily be enforced with the wrapper @signal_as_series_enforcer from [utils.py](utils.py). It accepts numerical (or convertible to float) lists, series, and tuples. When in doubt, wrap it up 