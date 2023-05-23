import numpy as np
import matplotlib.pyplot as plt

# Create a NumPy array
data = np.array([1, -2, -3, -3, 3, 4, 5, 5, -6, -6, 6, 6, 7, -8, -8, 9])

# Create x-axis values based on array indices
indices = np.arange(len(data))

# # Plot the bar chart
# plt.bar(indices, data)

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Bar Plot')

# # Show the plot
# plt.show()

# Set a seed for reproducibility (optional)
np.random.seed(42)

# Randomly choose indices for the subset
subset_indices = np.random.choice(len(data), size=5, replace=False)
subset_indices2 = np.random.choice(len(data), size=5, replace=False)

# Create the subset array
subset_array = data[subset_indices]
subset_array2 = data[subset_indices2]

print(subset_array)
print(subset_array2)
# Plot the bar chart
plt.plot(range(5), subset_array, label='subset_array', color='blue')
plt.plot(range(5), subset_array2, label='subset_array2', color='orange')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot')
plt.legend()

# Show the plot
plt.show()