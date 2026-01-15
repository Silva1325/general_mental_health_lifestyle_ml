import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import dataset
ds = pd.read_csv(r'synthetic_mental_health_dataset.csv')

# Indepentend variables and dependent variable
X = ds.iloc[:,:-2].values
y = ds.iloc[:,-2:].values

# Encoding categorical data
# Transforming diet_quality and weather columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [8,9])],remainder='passthrough')
X = ct.fit_transform(X)

# Seperate into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

class MultiOutputLinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiOutputLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
inputDim = X_train.shape[1] 
outputDim = y_train.shape[1] 
learningRate = 0.01
epochs = 1000
losses = []

model = MultiOutputLinearRegression(inputDim,outputDim)
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs,y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


weights = model.linear.weight.data.numpy()
biases = model.linear.bias.data.numpy()

# Using first output and first feature as example
trained_slope = weights[0, 0]  # First output, first feature
trained_intercept = biases[0]  # First output bias

# Creating ranges around the trained values
slope_range = np.linspace(trained_slope - 2, trained_slope + 2, 50)
intercept_range = np.linspace(trained_intercept - 2, trained_intercept + 2, 50)

# Creating meshgrid
slopes, intercepts = np.meshgrid(slope_range, intercept_range)
losses_grid = np.zeros_like(slopes)

# Evaluating the model
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)
    
    train_loss = criterion(train_pred, y_train)
    test_loss = criterion(test_pred, y_test)
    
    print(f'\nFinal Training Loss: {train_loss.item():.4f}')
    print(f'Final Test Loss: {test_loss.item():.4f}')

# Plot training loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot predictions for both outputs
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 0].numpy(), test_pred[:, 0].numpy(), alpha=0.5, label='Output 1')
plt.scatter(y_test[:, 1].numpy(), test_pred[:, 1].numpy(), alpha=0.5, label='Output 2')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual (Both Outputs)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot relationships between features and targets
# Get original feature names before encoding
feature_names = ['sleep_hours', 'screen_time', 'exercise_minutes', 'daily_pending_tasks',
                'interruptions', 'fatigue_level', 'social_hours', 'coffee_cups']

# Get the original dataset for plotting
ds_plot = pd.read_csv(r'synthetic_mental_health_dataset.csv')

# Create subplots for each feature vs both targets
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle('Feature Relationships with mood_score and stress_level', fontsize=16)

for idx, feature in enumerate(feature_names):
    row = idx // 2
    col = (idx % 2) * 2
    
    # Plot feature vs mood_score
    axes[row, col].scatter(ds_plot[feature], ds_plot['mood_score'], alpha=0.5, s=10)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('mood_score')
    axes[row, col].set_title(f'{feature} vs mood_score')
    axes[row, col].grid(True, alpha=0.3)
    
    # Plot feature vs stress_level
    axes[row, col+1].scatter(ds_plot[feature], ds_plot['stress_level'], alpha=0.5, s=10, color='orange')
    axes[row, col+1].set_xlabel(feature)
    axes[row, col+1].set_ylabel('stress_level')
    axes[row, col+1].set_title(f'{feature} vs stress_level')
    axes[row, col+1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot categorical features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Categorical Features vs Targets', fontsize=16)

# diet_quality vs mood_score
ds_plot.boxplot(column='mood_score', by='diet_quality', ax=axes[0, 0])
axes[0, 0].set_title('diet_quality vs mood_score')
axes[0, 0].set_xlabel('diet_quality')
axes[0, 0].set_ylabel('mood_score')

# diet_quality vs stress_level
ds_plot.boxplot(column='stress_level', by='diet_quality', ax=axes[0, 1])
axes[0, 1].set_title('diet_quality vs stress_level')
axes[0, 1].set_xlabel('diet_quality')
axes[0, 1].set_ylabel('stress_level')

# weather vs mood_score
ds_plot.boxplot(column='mood_score', by='weather', ax=axes[1, 0])
axes[1, 0].set_title('weather vs mood_score')
axes[1, 0].set_xlabel('weather')
axes[1, 0].set_ylabel('mood_score')

# weather vs stress_level
ds_plot.boxplot(column='stress_level', by='weather', ax=axes[1, 1])
axes[1, 1].set_title('weather vs stress_level')
axes[1, 1].set_xlabel('weather')
axes[1, 1].set_ylabel('stress_level')

plt.tight_layout()
plt.show()

with torch.no_grad():
    for i in range(len(slope_range)):
        for j in range(len(intercept_range)):
            # Temporarily set model parameters
            temp_weight = weights.copy()
            temp_bias = biases.copy()
            temp_weight[0, 0] = slopes[j, i]
            temp_bias[0] = intercepts[j, i]
            
            # Set temporary parameters
            model.linear.weight.data = torch.FloatTensor(temp_weight)
            model.linear.bias.data = torch.FloatTensor(temp_bias)
            
            # Calculate loss
            pred = model(X_train)
            loss = criterion(pred, y_train)
            losses_grid[j, i] = loss.item()
    
    # Restore original parameters
    model.linear.weight.data = torch.FloatTensor(weights)
    model.linear.bias.data = torch.FloatTensor(biases)

# 3D PLOT
fig4 = plt.figure(figsize=(14, 10))
ax = fig4.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(slopes, intercepts, losses_grid, cmap='viridis', alpha=0.8, edgecolor='none')

# Mark the trained parameters
ax.scatter([trained_slope], [trained_intercept], [losses[-1]], 
          color='red', s=100, marker='o', label='Trained Model')

# Labels and title
ax.set_xlabel('Slope (Weight[0,0])', fontsize=12)
ax.set_ylabel('Intercept (Bias[0])', fontsize=12)
ax.set_zlabel('Loss (MSE)', fontsize=12)
ax.set_title('3D Loss Surface: MSE as function of Slope and Intercept', fontsize=14, pad=20)

# Add colorbar
fig4.colorbar(surf, shrink=0.5, aspect=5)
ax.legend()
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# 2D CONTOUR PLOT
fig5 = plt.figure(figsize=(10, 8))
ax2 = plt.contourf(slopes, intercepts, losses_grid, levels=30, cmap='viridis')

# Add contour lines
contour_lines = plt.contour(slopes, intercepts, losses_grid, levels=15, colors='white', alpha=0.3, linewidths=0.5)
plt.clabel(contour_lines, inline=True, fontsize=8)

# Mark the trained parameters
plt.scatter([trained_slope], [trained_intercept], color='red', s=100, marker='o', 
           label='Trained Model', edgecolors='white', linewidths=2, zorder=5)

# Labels and title
plt.xlabel('Slope (Weight[0,0])', fontsize=12)
plt.ylabel('Intercept (Bias[0])', fontsize=12)
plt.title('2D Loss Contour: MSE as function of Slope and Intercept', fontsize=14)
plt.colorbar(ax2, label='Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()