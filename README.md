# EDA

Please see [eda.ipynb](eda.ipynb) for the main analysis

# Signal as pd Series
Most of functions in this repo treat signal as a Pandas Series:
* Values are RR intervals in milliseconds
* Index is time domain in milliseconds

These formats can very easily be enforced with the wrapper @signal_as_series_enforcer from [utils.py](utils.py). It accepts numerical (or convertible to float) lists, series, and tuples. When in doubt, wrap it up 