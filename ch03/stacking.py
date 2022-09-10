# 1 We start with the libraries import
from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 2 Laod the boston dataset (it's biased, be careful)
data = Dataset.load_boston()

# 3 Split the data
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=2)

# 4 Let's create the dataset
Data = Dataset(X_train,y_train,X_test)

# 5 Now we can build the 2 models that we will use in the stacking procedure
RfModel = Regressor(dataset=Data, estimator=RandomForestRegressor, parameters={'n_estimators': 50},name='rf')
LRModel = Regressor(dataset=Data, estimator=LinearRegression, parameters={'normalize': True},name='lr')

# 6 It's time to stack these models
Pipeline = ModelsPipeline(RfModel, LRModel)
StackModel = Pipeline.stack(k=10, seed=2)

# 7 Now we will train a LinearRegression model on stacked data
Stacker = Regressor(dataset=StackModel, estimator=LinearRegression)

# 8 Finally, we calculate the results to validate the model
Results = Stacker.predict()
Results = Stacker.validate(k=10,scorer=mean_absolute_error)