# 1 Let's make the basic imports
import pyltr

# 2 We will load the data contained in the Letor dataset that's already
# provided
with open('ch06/train.txt') as trainfile, open('ch06/vali.txt') as valifile, open('ch06/test.txt') as testfile:
    TrainX, Trainy, Trainqids, _ = pyltr.data.letor.read_dataset(trainfile)
    ValX, Valy, Valqids, _ = pyltr.data.letor.read_dataset(valifile)
    TestX, Testy, Testqids, _ = pyltr.data.letor.read_dataset(testfile)
    metric = pyltr.metrics.NDCG(k=10)

# 3 Let's now perform a validation of the data
monitor = pyltr.models.monitors.ValidationMonitor(ValX, Valy, Valqids, metric=metric, stop_after=250)

# 4 We will build the model
model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

# 5 Now we can fit the model using the text data
model.fit(TestX, Testy, Testqids, monitor=monitor)

# 6 Next, we can predict the data
Testpred = model.predict(TestX)

# 7 Finally we print the results
print('Random ranking:', metric.calc_mean_random(Testqids, Testy))
print('Our model:', metric.calc_mean(Testqids, Testy, Testpred))

