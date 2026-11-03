import numpy as np

class LinearRegression:
	def __init__(self, learning_rate=0.0001, n_iters=1000, method="gradient_descent"):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.method = method

		# Model parameters
		self.weights = None
		self.bias = None

		# Training history
		self.loss_history = []

		# Model information
		self.n_samples = None
		self.n_features = None


	def initialize_parameters(self, n_features):
		self.weights = np.zeros(n_features)
		self.bias = 0.0

	def forward_propagation(self, X):
		y_pred = X @ self.weights + self.bias
		return y_pred

	def compute_mse_loss(self, y_true, y_pred):
		mse = np.mean((y_pred - y_true) ** 2)
		return mse / 2

	def compute_gradients_descent(self, X, y_true, y_pred):
		m = X.shape[0]
		error = y_pred - y_true
		dw = (X.T @ error) / m
		db = np.sum(error) / m
		return dw, db

	def gradients_descent(self, X, y):
		for epoch in range(self.n_iters):
			# Forward propagation
			y_pred = self.forward_propagation(X)

			# Calculate loss
			loss = self.compute_mse_loss(y, y_pred)
			self.loss_history.append(loss)

			# Calculate gradients
			dw, db = self.compute_gradients_descent(X, y, y_pred)

			# Update parameters
			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

	def normal_equation(self, X, y):
		# Add bias column
		X_bias = np.c_[np.ones((X.shape[0], 1)), X]
		# pinv is more robust when X.T @ X is singular
		theta = np.linalg.pinv(X_bias) @ y
		self.bias = theta[0]
		self.weights = theta[1:]
		# Record loss
		y_pred = self.forward_propagation(X)
		loss = self.compute_mse_loss(y, y_pred)
		self.loss_history.append(loss)

	def fit(self, X, y):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float)

		# Ensure X is 2D
		if X.ndim == 1:
			X = X.reshape(-1,1)

		# Ensure y is 1D
		y = y.reshape(-1)

		# Validate input
		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have the same number of samples")

		if X.shape[0] == 0:
			raise ValueError("Training data cannot be empty")

		if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
			raise ValueError("X and y must contain finite values")

		self.n_samples, self.n_features = X.shape

		# Initialize parameters
		self.initialize_parameters(self.n_features)

		self.loss_history = []

		# Select optimize method
		if self.method == "gradient_descent":
			self.gradients_descent(X, y)
		elif self.method == "normal_equation":
			self.normal_equation(X, y)
		else:
			raise ValueError("Method must be gradient_descent or normal_equation")
		return self

	def predict(self, X):
		if self.weights is None:
			raise ValueError("Model has not been trained yet")

		X = np.asarray(X, dtype = float)
		if X.ndim == 1:
			X = X.reshape(-1, self.n_features)
		if X.shape[1] != self.n_features:
			raise ValueError("Number of features does not match")

		return self.forward_propagation(X)

	@staticmethod
	def mse(y_true, y_pred):
		return np.mean((y_true - y_pred) ** 2)

	@staticmethod
	def rmse(y_true, y_pred):
		return np.sqrt(np.mean((y_true - y_pred) ** 2))

	@staticmethod
	def mae(y_true, y_pred):
		return np.mean(np.abs(y_true - y_pred))

	@staticmethod
	def r2_score(y_true, y_pred):
		ss_res = np.sum((y_true - y_pred) ** 2)
		ss_tot = np.sum(y_true - np.mean(y_true) ** 2)
		if ss_tot == 0:
			return 1.0 if ss_res == 0 else 0.0
		return 1 - (ss_res / ss_tot)

	def model_information(self):
		return {
				"weights": self.weights,
				"bias": self.bias,
				"n_samples": self.n_samples,
				"n_features": self.n_features
				}