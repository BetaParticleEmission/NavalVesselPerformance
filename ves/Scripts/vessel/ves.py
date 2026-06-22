import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import seaborn as sns 
import configparser

vf = pd.read_csv(r'C:\Users\isaac\Documents\Personal_Projects\AI\navy_vessel\Scripts\vessel\data.csv')

# Feature handling
vf.drop(columns="index", inplace=True)

# Missing values
print(f"Missing Values in Vessels:\n{vf.isnull().sum()}") 

# Duplicate values 
print(f"Number of duplicates:{vf.duplicated().sum()}")

# Outlier Detection 
Q1 = np.percentile(vf,25)
Q3 = np.percentile(vf, 75)

iqr = Q3 - Q1 

lower= Q1 - 1.5*iqr 
upper= Q3 + 1.5*iqr

cols = list(vf.columns)

outliers = {}
for col in cols: 
    outliers[col] = vf[(vf[col] > upper) | (vf[col] < lower)].sum()
    print(f"{outliers}")

sns.boxplot(vf,orient="h", color="blue")
plt.title("Outlier Visual") 

# Feature Cleaning 
from janitor import clean_names

vf = clean_names(vf)
vf.head()

# Feature Creation and Selection 

vf["v_ms"] = vf["ship_speed_v_"] * 0.5144
vf["Velocity_mf"] = vf["v_ms"] / vf["fuel_flow_mf_[kg_s]_"]

X = vf.drop("Velocity_mf", axis = 1)
y = vf["Velocity_mf"]

from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X,y)
mi_mask = mi_scores > 0.05
X_fil = X.loc[:,mi_mask]

## Multicollinearity filter

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_fil = add_constant(X_fil)
def compute_vif(X): 
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) 
                  for i in range(X.shape[1])]
    return vif.sort_values('VIF', ascending=False)

vif = compute_vif(X_fil) ### High VIF, use Ridge Regression


df1 = vf.drop(columns = ["gt_compressor_inlet_air_temperature_t1_[c]_", "gt_compressor_inlet_air_pressure_p1_[bar]_"])
corr_matrix = df1.corr()
corr_matrix["Velocity_mf"].sort_values(ascending=True)

sns.heatmap(corr_matrix[['Velocity_mf']].sort_values(by='Velocity_mf', ascending=False), annot=True)

# use VIF and correlation to select possible features 
df1.rename({"hight_pressure_hp_turbine_exit_temperature_t48_": "height_pressure_hp_turbine_exit_temp"}, inplace=True)
X = df1[["turbine_injecton_control_tic_[%]_", "hight_pressure_hp_turbine_exit_temperature_t48_[c]_","gt_rate_of_revolutions_gtn_[rpm]_", "lever_position_"]]
y = df1["Velocity_mf"]

df_fin = pd.concat([X, y.to_frame()], axis=1)

# EDA

# Modeling 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

## Apply scaling 

scaler = StandardScaler() 
scaler.fit_transform(X,y) 

compute_vif(df_fin)

## Train-test-split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0) 
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)

from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

print(f"RSME: {root_mean_squared_error(y_test, y_pred)}")
print(f"r2 score: {r2_score(y_test, y_pred)}")

# Test Model Assumptions (i.e Linearity, Predictor Independence, Homeoscedasticity)

# fitted and residuals for training data

import statsmodels
y_train_pred = ridge.predict(x_train)
train_residuals = y_train - y_train_pred

train_df = pd.DataFrame({
    "Actual": y_train,
    "Fitted": y_train_pred,
    "Residuals": train_residuals

})


from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import boxcox, boxcox_normplot

norm_value = shapiro(train_residuals)
print(f"Shapiro-test value: {norm_value.statistic}")
print(f"Shapiro-test p-value: {norm_value.pvalue}")

probplot(train_residuals, dist="norm", plot=plt)
plt.title("QQplot of Training Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Residuals")
plt.show()

y_transformed, best_lambda = boxcox(y)
print(f"Optimal Lambda: {best_lambda: .4f}")

plt.hist(y, bins = 50, color="steelblue", edgecolor="white")

transformed_y, best_lambda = boxcox(y)
print(f"Optimal lambda:{best_lambda:.4f}")

plt.hist(transformed_y, color="steelblue", bins=50, edgecolor="white")

x_train1, x_test1, y_train1, y_test1 = train_test_split(X,transformed_y, test_size=0.2, random_state=42)

y_train_pred1 = ridge.predict(x_train1)
train_residuals1 = y_train1 - y_train_pred1

train_df1 = pd.DataFrame({
    "Actual": y_train1,
    "Fitted": y_train_pred1,
    "Residuals": train_residuals1

})

probplot(train_residuals1, dist="norm", plot=plt)
plt.title("QQplot of Training Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Residuals")
plt.show()

test, p = shapiro(train_residuals1)
print("test:", test)
print("p-value:", p)

# Revamped Ridge model (S: Recursive Feature Elimination + Ridge + CV)


from sklearn.feature_selection import RFECV 
from sklearn.linear_model import Ridge 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits = 5, shuffle=True, random_state=42)
x_train1, x_test1, y_train1, y_test1 = train_test_split(X,y, test_size=0.2, random_state=42)

selector = RFECV(
    Ridge(alpha=1.0),
    step=1,
    cv=cv,
    scoring= "neg_mean_squared_error"
)

y_transformed, best_lambda = boxcox(y_train1)
selector.fit(x_train1, y_transformed)
print(f"Optimal number of features: {selector.n_features_}")
X_train_rfe = selector.transform(x_train1)
print(f"Best Features: {list(x_train1.columns)}")

ridge_vamp = Ridge(alpha=1.0)

scores = cross_val_score(ridge_vamp, X_train_rfe, y_transformed, cv = cv)
print(f"R2 scores:{scores}")
print(f"Mean: {scores.mean(): .4f}")
print(f"Std: {scores.std(): .4f}") # Good std

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    ridge_vamp, X_train_rfe, y_transformed, 
    cv = cv,
    train_sizes = np.linspace(0.1,1.0, 10)
)

plt.plot(train_sizes, -train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, -test_scores.mean(axis=1), label='Test')
plt.xlabel("Training Size")
plt.ylabel("RMSE")
plt.legend()
plt.title("Learning Curve")
plt.show()


import streamlit as st 

st.title("Naval Vessel Performance")

class Learning():
    def __init__(self, model, xtrain, y_transformed, cv):

        self.model = model 
        self.xtrain = xtrain 
        self.y_transformed = y_transformed 
        self.cv = cv 
    
    def learning_plot(self): 
    
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.xtrain, self.y_transformed, 
            cv = self.cv,
            train_sizes = np.linspace(0.1,1.0, 10)
        )

        
        plt.plot(train_sizes, -train_scores.mean(axis=1), label='Train')
        plt.plot(train_sizes, -test_scores.mean(axis=1), label='Test')
        plt.xlabel("Training Size")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title("Learning Curve")
        plt.show()
    
    def feature_importance(self): 

        from sklearn.inspection import permutation_importance

        result = permutation_importance(self.model.fit(self.xtrain, self.y_transformed), self.xtrain, self.y_transformed, n_repeats=10, random_state=42)

        # Create DataFrame
        perm_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)

        return perm_importance_df

st.write("List key terms:")
levbutton = st.checkbox("Lever")

if levbutton: 
   
    st.write("**Lever position**: Refers to the distance between center of gravity and vertical line of action for the new center of buoyancy upon heeling.")



st.write("Learning Curve:")
pltbutton = st.button("See the Learning Curve",type="primary")
fig, ax = plt.subplots()

if pltbutton: 
    l = Learning(model=ridge_vamp, xtrain=X_train_rfe, y_transformed=y_transformed, cv=cv).learning_plot()
    st.pyplot(fig, l)

st.write("List Feature Importances:")

feabutton = st.button("See the Feature Importances", type="primary")

if feabutton: 
    f = Learning(model=ridge_vamp, xtrain=X_train_rfe, y_transformed=y_transformed, cv=cv).feature_importance()
    st.dataframe(f)
    st.write("We see that the most important features (in-order) are: lever position, turbine exit temperature, gas turbine rate of revolution, then turbine injection control.")
  