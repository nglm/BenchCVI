BenchCVI
===============================================================================

Benchmarking of CVIs (Cluster Validity Indices) compatible with static and times series data and clustering methods. The package can also handle both clustering methods that mostly rely on the parameter k (number of clusters) and those that do not require it and evaluate CVIs for both types of clustering methods.

BenchCVI is notably compatible with:

- [PyCVI](https://github.com/nglm/pycvi), where some CVIs are implemented in Python, compatible with static and time-series data, and making it possible to use sklearn clustering methods with time series metrics (for clustering methods that accept a custom metric, such as OPTICS, AgglomerativeClustering, HDBSCAN, etc.)
- [sklearn](https://scikit-learn.org/stable/index.html), where clustering methods for static data are implemented
- [kmedoids](https://python-kmedoids.readthedocs.io/en/latest/), a sklearn-like implementation of the KMedoids algorithm for static data
- [aeon](https://www.aeon-toolkit.org/en/stable/) and [sktime](https://www.sktime.net/docs/users/), where clustering methods for time series data are implemented, as well as time-series distances and average functions.
