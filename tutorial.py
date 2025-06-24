import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic 2D data
np.random.seed(0)
x = np.random.rand(300, 2)
y = np.array(x[:, 0] + x[:, 1] > 1, dtype=int)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=100, verbose=0)

# Create meshgrid for visualization
xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = model.predict(grid, verbose=0).reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.6, cmap='coolwarm')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Model's Decision Boundary")
plt.show()
