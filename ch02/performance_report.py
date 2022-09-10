# 1 Add the usual reports
from sklearn.metrics import classification_report
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true, y_pred, target_names=target_names))

# 2 Run the code and see
# Instead of computing these metrics separately, you can directly
# use the preceding function to extract those statistics from your model.