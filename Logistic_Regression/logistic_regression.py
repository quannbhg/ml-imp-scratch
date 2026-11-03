import numpy as np

class LogisticRegression:
	def __init__(self, learning_rate = 0.0001, n_iters = 1000, threshold = 0.5, _lambda = 0.0):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.threshold = threshold
		self._lambda = _lambda

		# Model parameters
		self.weights = None
		self.bias = None

		# Training history
		loss_history = []

		# Dataset information
		self.n_samples = None
		self.n_features = None

	def initialize_parameters(self, n_features):
		self.weights = np.zeros(n_features)
		self.bias = 0.0

	@staticmethod
	def sigmoid(z):
		z = np.clip(z, -500, 500)
		return 1.0 / (1.0 + np.exp(-z))

	def forward_propagation(self, X):
		z = X @ self.weights + self.bias
		y_pred = self.sigmoid(z)
		return y_pred

	def compute_bce_loss(self, y_true, y_pred):
		epsilon = 1e-15
		y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
		bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
		# L2 Regularization
		l2 = (self._lambda / (2 * self.n_samples)) * np.sum(self.weights ** 2)
		return bce + l2

	def compute_gradients(self, X, y_true, y_pred):
		m = X.shape[0]
		error = y_pred - y_true
		dw = (X.T @ error / m) + (self._lambda / m) * self.weights
		db = np.sum(error) / m
		return dw, db

	def gradients_descent(self, X, y):
		for epoch in range(self.n_iters):
			# Forward propagation
			y_pred = self.forward_propagation(X)

			# Calculate loss
			loss = self.compute_bce_loss(y, y_pred)
			self.loss_history.append(loss)

			# Calculate gradients
			dw, db = self.compute_gradients(X, y, y_pred)

			# Update parameters
			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

	def fit(self, X, y):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float)

		if X.ndim == 1:
			X = X.reshape(-1, 1)

		if X.shape[0] != y.shape[0]:
			raise ValueError("X and y must have same number of samples")

		if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
			raise ValueError("X and y must contain finite values")

		if not np.all(np.isin(y, [0, 1])):
			raise ValueError("Labels must be 0 or 1")

		if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
			raise ValueError("Learning rate must be positive")

		if self.n_iters <= 0:
			raise ValueError("n_iters must be positive")

		if not 0 < self.threshold < 1:
			raise ValueError("Threshold must be between 0 and 1")

		if self._lambda < 0:
			raise ValueError("_lambda must be non-negative")

		self.n_samples, self.n_features = X.shape

		# Initialize parameters
		self.initialize_parameters(self.n_features)

		self.loss_history = []

		# Training model
		self.gradients_descent(X, y)

		return self

	def predict_probability(self, X):
		if self.weights is None:
			raise ValueError("Model has not been trained yet")

		X = np.asarray(X, dtype=float)

		if X.ndim == 1:
			X = X.reshape(-1, 1)

		if X.shape[1] != self.n_features:
			raise ValueError("X and y must have same number of samples")

		return self.forward_propagation(X)

	def predict(self, X):
		probability = self.predict_probability(X)

		return (probability >= self.threshold).astype(int)

	@staticmethod
	def acccuracy(y_true, y_pred):
		return np.mean(y_true == y_pred)

	@staticmethod
	def confusion_matrix(y_true, y_pred):
		# True Negative
		tn = np.sum((y_true == 0) & (y_pred == 0))

		# False Positive
		fp = np.sum((y_true == 0) & (y_pred == 1))

		# False Negative
		fn = np.sum((y_true == 1) & (y_pred == 0))

		# True Positive
		tp = np.sum((y_true == 1) & (y_pred == 1))

		return np.array([[tn, fp], [fn, tp]])

	@staticmethod
	def precision(y_true, y_pred):
		tp = np.sum((y_true == 1) & (y_pred == 1))
		fp = np.sum((y_true == 0) & (y_pred == 1))

		return tp / (tp + fp) if tp + fp > 0 else 0.0

	@staticmethod
	def recall(y_true, y_pred):
		tp = np.sum((y_true == 1) & (y_pred == 1))
		fn = np.sum((y_true == 1) & (y_pred == 0))

		return tp / (tp + fn) if tp + fn > 0 else 0.0

	@staticmethod
	def f1_score(y_true, y_pred):
		p = LogisticRegression.precision(y_true, y_pred)
		r = LogisticRegression.recall(y_true, y_pred)\

		return 2 * p * r / (p + r) if p + r > 0 else 0.0

	def model_information(self):
		return {
				"weights": self.weights,
				"bias": self.bias,
				"n_samples": self.n_samples,
				"n_features": self.n_features
				}