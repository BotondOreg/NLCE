## NLCE Calculations
Numerical Linked-Cluster Expansion algorithm implemented for the Fermi-Hubbard model.

## In this branch
Need to modify a file in scipy to make sure the code doesn't break.
Do this at your own risk but if you are using a virtual environment, it should be fairly safe.
Will use the local version at some point.

The necessary change:
Add the following lines to ```envs/<environment-name>/Lib/site-packages/scipy/sparse/linalg/_expm_multiply.py```:
```
# EDIT
X = np.empty((nsamples,), dtype=object) # dtype is a sparse matrix
```

after line 595:
```
X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
```

A modified file is also there, you can copy that.
