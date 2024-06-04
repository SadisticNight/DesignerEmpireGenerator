import numpy as np
import itertools
import time

NUM_CELDAS = 200
radio = 50
x_centro, y_centro = 100, 100

# Método numpy
start_time = time.time()
x_range = np.arange(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1))
y_range = np.arange(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1))
x_grid, y_grid = np.meshgrid(x_range, y_range, indexing='ij')
mask = (x_grid - x_centro) ** 2 + (y_grid - y_centro) ** 2 <= radio ** 2
area_numpy = set(zip(x_grid[mask].ravel(), y_grid[mask].ravel()))
end_time = time.time()
print(f"Numpy took {end_time - start_time:.6f} seconds")

# Método itertools.product
start_time = time.time()
range_x = range(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1))
range_y = range(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1))
area_itertools = {
    (x, y)
    for x, y in itertools.product(range_x, range_y)
    if (x - x_centro) ** 2 + (y - y_centro) ** 2 <= radio ** 2
}
end_time = time.time()
print(f"itertools.product took {end_time - start_time:.6f} seconds")

# Método combinado
start_time = time.time()
x_range = np.arange(max(0, x_centro - radio), min(NUM_CELDAS, x_centro + radio + 1))
y_range = np.arange(max(0, y_centro - radio), min(NUM_CELDAS, y_centro + radio + 1))
x_grid, y_grid = np.meshgrid(x_range, y_range, indexing='ij')
area_combinada = {
    (x, y)
    for x, y in zip(x_grid.ravel(), y_grid.ravel())
    if (x - x_centro) ** 2 + (y - y_centro) ** 2 <= radio ** 2
}
end_time = time.time()
print(f"Combinación numpy y itertools took {end_time - start_time:.6f} seconds")
