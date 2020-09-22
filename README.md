Requirement:
Tested under Python 3.7.4
Please install by pip if the following libraries are missing:
pandas, numpy, jsonpickle, sklearn, tqdm, matplotlib

Source code:
Entry files are the ones starting with "exp-", each represents a different experiment mentioned in the paper. You can find the configurations in each file.

Notebook:
A couple Jupyter Notebooks are included under the notebook folder, we use them for plotting.

Data:
The synthetic, Intel wiress and Instacart dataset are included under the data folder.
The NYC Taxi can be downloaded at https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
The taxi dataset require some preprocessing which you can apply by notebook/explore.ipynb

Others:
verdictdb.md contains some notes/tips for setting up the VerdictDB experiment.
