import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]
    
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
    
class MulticlassLogisticRegression():
    def __init__(self, learning_rate=0.01, n_iters=1000, batch_size=32, lambda_reg=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg  # L2 regularization
        self.weights = None
        self.bias = None
        self.best_weights = None
        self.best_bias = None
        self.classes_ = None
        self.scaler = None
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=1):   
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Inisialisasi Weights
        self.weights = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / (n_features + n_classes))
        self.bias = np.zeros(n_classes)
        
        y_one_hot = one_hot(y, n_classes)

        self.best_val_accuracy = 0
        
        # Training loop
        for i in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            batch_losses = []
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:min(start_idx + self.batch_size, n_samples)]
                X_batch = X[batch_indices]
                y_batch_one_hot = y_one_hot[batch_indices]
                
                # Forward Pass
                linear_output = np.dot(X_batch, self.weights) + self.bias
                probs = softmax(linear_output)
                
                # Compute loss (Cross-Entropy)
                batch_loss = -np.mean(np.sum(y_batch_one_hot * np.log(probs + 1e-10), axis=1))
                batch_loss += 0.5 * self.lambda_reg * np.sum(self.weights**2)  # L2 penalty
                batch_losses.append(batch_loss)
                
                # Gradient dengan Regularisasi L2
                dw = np.dot(X_batch.T, (probs - y_batch_one_hot)) / len(batch_indices)
                dw += self.lambda_reg * self.weights  # L2 regularization
                db = np.sum(probs - y_batch_one_hot, axis=0) / len(batch_indices)
                
                # Update weights dengan learning rate
                self.weights = self.weights - self.lr * dw
                self.bias = self.bias - self.lr * db
            
            # Train Accuracy
            train_preds = self.predict(X)
            train_acc = accuracy(y, train_preds)
            train_loss = np.mean(batch_losses)
            
            # Validasi Matrix Calculation
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_preds = np.argmax(val_probs, axis=1)
                val_accuracy = accuracy(y_val, val_preds)
                
                # Menghitung Validation Loss
                y_val_one_hot = one_hot(y_val, n_classes)
                val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_probs + 1e-10), axis=1))
                val_loss += 0.5 * self.lambda_reg * np.sum(self.weights**2)
                
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias.copy()
            
        # Menggunakan Best Weights
        if X_val is None and y_val is None:
            self.best_weights = self.weights.copy()
            self.best_bias = self.bias.copy()
            
        self.weights = self.best_weights
        self.bias = self.best_bias
            
        return self
    
    def predict_proba(self, X):
        # Prediksi Kelas probabilitas
        linear_output = np.dot(X, self.weights) + self.bias
        return softmax(linear_output)
        
    def predict(self, X):
        # Prediksi Kelas Label
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y, verbose=1):
        # Evaluasi Performa Model
        y_pred = self.predict(X)
        acc = accuracy(y, y_pred)
        
        if verbose > 0:
            print(f"Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y, y_pred))
            
        return acc