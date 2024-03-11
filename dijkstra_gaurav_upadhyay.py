import numpy as np
import heapq
import math
import cv2
from google.colab.patches import cv2_imshow

# Define the map dimensions
map_width = 1200
map_height = 500

# Create a blank canvas with white background
map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

# Draw border
border_color = (50, 250, 0)  # Black color for border
cv2.rectangle(map_img, (0, 0), (map_width+1, map_height+1), border_color, 5)

# Draw obstacles and walls on the image
obstacle_color = (0, 0, 0)  # Black color for obstacles
obs1 = cv2.rectangle(map_img, (100, 0), (175, 400), obstacle_color, 1)
obs2 = cv2.rectangle(map_img, (275, 100), (350, 500), obstacle_color, 1)
obs3 = cv2.rectangle(map_img, (900, 50), (1020, 125), obstacle_color, 1)
obs4 = cv2.rectangle(map_img, (1020, 50), (1100, 450), obstacle_color, 1)
obs5 = cv2.rectangle(map_img, (900, 375), (1020, 450), obstacle_color, 1)

# Define center and side length of the hexagon
center = (650, 250)  # Center coordinates of the canvas
side_length = 150  # Length of each side of the hexagon

# Calculate the coordinates of the vertices of the hexagon
vertices = []
for i in range(6):
    angle_deg = 60 * i - 30
    angle_rad = np.deg2rad(angle_deg)
    x = int(center[0] + side_length * np.cos(angle_rad))
    y = int(center[1] + side_length * np.sin(angle_rad))
    vertices.append((x, y))

# Draw the hexagon on the canvas
obs6 = cv2.polylines(map_img, [np.array(vertices)], isClosed=True, color=obstacle_color, thickness=1)

# Define start and goal points
start = (50, 50)
goal = (50, 400)
video_name = "exploration_video.avi"  # Define the video name here

# Function to check if a node is valid (inside the grid and not an obstacle)
def is_valid(node):
    x, y = node
    return 0 <= x < map_width and 0 <= y < map_height and map_img[y, x, 0] != 0

# Function to generate the graph considering obstacles
def generate_graph():
    graph = {}
    for i in range(map_width):
        for j in range(map_height):
            if is_valid((i, j)):
                neighbors = {(i + dx, j + dy) for dx in range(-1, 2) for dy in range(-1, 2)}
                neighbors.discard((i, j))  # Remove the current node from neighbors
                neighbors = {n for n in neighbors if is_valid(n)}
                graph[(i, j)] = neighbors
    return graph

# Function to calculate the costs at each node
def calculate_costs(graph):
    costs = {}
    for node, neighbors in graph.items():
        costs[node] = {}
        for neighbor in neighbors:
            dx, dy = neighbor[0] - node[0], neighbor[1] - node[1]
            if dx != 0 and dy != 0:
                costs[node][neighbor] = math.sqrt(2)
            else:
                costs[node][neighbor] = 1
    return costs

# Dijkstra's algorithm to find the shortest path
def dijkstra(graph, start, goal, video_name):
    visited = set()
    distances = {node: math.inf for node in graph}
    distances[start] = 0
    heap = [(0, start)]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (map_width, map_height))

    while heap:
        dist, node = heapq.heappop(heap)
        if node == goal:
            break
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            new_distance = dist + costs[node][neighbor]
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))
        # Record the explored node
        map_img[node[1], node[0]] = (0, 0, 255)  # Mark explored node as green
        out.write(map_img)  # Save frame as image

    # Release the VideoWriter object
    out.release()

    return distances

# Backtrack to find the path from goal to start
def backtrack(graph, start, goal, distances):
    path = [goal]
    while path[-1] != start:
        neighbors = [(n, distances[n]) for n in graph[path[-1]] if distances[n] < math.inf]
        next_node = min(neighbors, key=lambda x: x[1])[0]
        path.append(next_node)
    return path[::-1]

# Main function to run the algorithm
def run_dijkstra(start, goal, video_name):
    graph = generate_graph()
    if goal not in graph:
        print("Goal point is unreachable or invalid.")
        return None
    global costs
    costs = calculate_costs(graph)
    distances = dijkstra(graph, start, goal, video_name)
    path = backtrack(graph, start, goal, distances)
    return path

# Function to draw the path on the grid
def draw_path(path, map_img):
    if path is None:
        print("No valid path found.")
        return map_img

    # Draw the path on the map image
    path_color = (255, 0, 0)  # Red color for the path
    for node in path:
        cv2.circle(map_img, node, 2, path_color, -1)

    return map_img

# Example usage
if __name__ == "__main__":
    path = run_dijkstra(start, goal, video_name)
    if path:
        path_image = draw_path(path, map_img.copy())
        cv2_imshow(path_image)
        cv2.waitKey(0)
    else:
        print("Path computation failed.")
