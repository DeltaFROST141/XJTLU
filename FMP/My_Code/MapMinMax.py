import numpy as np

def mapminmax(x, y_min=0, y_max=1):
    x_min = x.min()
    x_max = x.max()

    y = (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min

    return y, (x_min, x_max, y_min, y_max)

def reverse_mapminmax(y, params):
    x_min, x_max, y_min, y_max = params

    x = (y - y_min) * (x_max - x_min) / (y_max - y_min) + x_min

    return x

# Test
data = np.array([1, 2, 3, 4, 5])
scaled_data, params = mapminmax(data)
original_data = reverse_mapminmax(scaled_data, params)

print(scaled_data)
print(original_data)
