# Multiple Kernel Least Squares Support Vector Regression (MK LS SVR)

Import package
-----------------------------------
```python
import mk_ls_svr
```
Select kernel set
-----------------------------------
```python
import mk_ls_svr
kernel_list = []
for i in range(0, 10):
    kernel_list.append(mk_ls_svr.Kernel("rbf", [kernel_params[i]]))
regr_estimator = mk_ls_svr.MKLSSVR(kernel_list, c=reg_param)
```

Fit regressor
-----------------------------------
```python
datafile = "Datasets\\test\\test_0.1_30.xlsx"
data = pd.read_excel(datafile, header=0)
X_data = data.drop("y", axis=1).as_matrix()
X_test = pd.read_excel(datafile, header=0).as_matrix()
Y_data = np.array(data["y"])

regr_estimator.fit(X_data, Y_data)

```

Predict
-----------------------------------
```python
y_est = regr_estimator.predict(X_test)
```
