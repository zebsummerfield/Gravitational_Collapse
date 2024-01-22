import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation

# Function to be applied in parallel
def square(x):
    return x**2

# Wrapper function for parallel mapping
def parallel_map_wrapper(data, func, num_processes=4):
    with Pool(processes=num_processes) as pool:
        result = pool.map(func, data)
    return result

# Function to update the plot during animation
def update(frame):
    ax.clear()
    x_values = list(range(frame + 1))
    y_values = parallel_map_wrapper(x_values, square)
    ax.plot(x_values, y_values, marker='o')
    ax.set_title(f'Frame: {frame}')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')

if __name__ == "__main__":
    num_frames = 10
    fig, ax = plt.subplots()
    
    # Create an animation
    animation = FuncAnimation(fig, update, frames=num_frames, interval=500)

    plt.show()