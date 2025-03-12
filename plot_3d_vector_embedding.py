import matplotlib.pyplot as plt
import torch
def plot3D_vector_embedding(input_data,words):

    x_coords = input_data[:, 0].numpy()
    y_coords = input_data[:, 1].numpy()
    z_coords = input_data[:, 2].numpy()

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point and annotate with corresponding word
    for x, y, z, word in zip(x_coords, y_coords, z_coords, words):
        ax.scatter(x, y, z)
        ax.text(x, y, z, word, fontsize=10)
    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.title('3D Plot of Word Embeddings')
    plt.show()

def plot_vector_embedding_from_origin(input_data,words):
    # Create 3D plot with vectors from origin to each point, using different colors
    x_coords = input_data[:, 0].numpy()
    y_coords = input_data[:, 1].numpy()
    z_coords = input_data[:, 2].numpy()
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = "3d")

    # Define a list of colors for the vectors
    colors = ['brown', 'g', 'b', 'c', 'm', 'y', 'r']
    # Plot each vector with a different color and annotate with the corresponding word
    for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
        # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
        ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
        ax.text(x, y, z, word, fontsize=10, color=color)
    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set plot limits to keep arrows within the plot boundaries
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.title('3D Plot of Word Embeddings with Colored Vectors')
    plt.show()



## TEST

input_data = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
   [0.4419, 0.6515, 0.5683]]
)

words = ['Your', 'journey', 'starts', 'with', 'one', 'step', 'journey-context']

#plot3D_vector_embedding(input_data,words)
plot_vector_embedding_from_origin(input_data,words)