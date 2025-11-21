from ml_assignment_2.modeling.neural_network import neural_network

adam_nn = neural_network(solver="adam", random_state=0)
sgd_nn  = neural_network(solver="sgd", random_state=0)

adam_nn.train_optuna(X_train, y_train)  # or .model = adam_nn.classifier().fit(...)
sgd_nn.train_optuna(X_train, y_train)

results_adam = adam_nn.evaluate(X_train, X_test, y_train, y_test)
results_sgd  = sgd_nn.evaluate(X_train, X_test, y_train, y_test)