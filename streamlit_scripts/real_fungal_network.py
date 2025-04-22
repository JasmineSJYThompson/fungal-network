from scipy.io import loadmat
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import pyvista as pv
import streamlit as st

from stpyvista import stpyvista

import platform
import os

FOLDER = "/data/fungal_networks/"
#MAT_FILENAMES = ["Pp_M_Tokyo_U_N_26h_1.mat", "Pp_M_Tokyo_U_N_26h_2.mat"
NETWORK_DRAWING_LAYOUTS = ["spring_layout", "kamada_kawai_layout", "none"] #Added none option

#(platform.system() != "Linux")
if  "IS_XVFB_RUNNING" not in st.session_state:
    pv.start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

pv.global_theme.allow_empty_mesh = True

def get_mat_filenames():
    mat_filenames = os.listdir(os.getcwd() + FOLDER)
    return mat_filenames

def load_mat_data(mat_selection):
    data = loadmat(f"{os.getcwd() + FOLDER}{mat_selection}")
    A = data["A"].tocoo()
    coordinates = data["coordinates"]

    nodes = pd.DataFrame({"id": range(coordinates.shape[0]), "x": coordinates[:, 0], "y": coordinates[:, 1]})
    edges = pd.DataFrame({"source": A.row, "target": A.col, "weight": A.data})
    # Filters out very tiny weights for the purpose of a better visualisation
    #edges = edges[edges["weight"] >= 1]

    return A, nodes, edges, coordinates

def get_original_coordinates(coordinates, is_3d=False):
    if is_3d:
        mu = np.array(coordinates).mean()
        sigma = np.array(coordinates).std()
        coords_3D = pd.DataFrame(coordinates).to_dict("index")
        coords_3D = {key: [value[0], value[1], np.random.normal(mu, sigma)] for key, value in coords_3D.items()}
        return coords_3D
    else:
        coords_2D = pd.DataFrame(coordinates).to_dict("index")
        coords_2D = {key: [value[0], value[1]] for key, value in coords_2D.items()}
        return coords_2D

def construct_graph(nodes, edges):
    G = nx.Graph()
    for node in nodes.to_dict("records"):
        G.add_node(node["id"])
    for edge in edges.to_dict("records"):
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    return G


def get_edge_weights(G):
    edge_weights = [G[u][v]["weight"] * 1.5 for u, v in G.edges]
    return edge_weights

def get_node_positions(network_drawing_layout, G, coords, is_3d=False):
    if network_drawing_layout == "spring_layout":
        if is_3d:
            pos = nx.spring_layout(G, seed=42, pos=coords, weight=None, dim=3)
        else:
            pos = nx.spring_layout(G, seed=42, pos=coords, weight=None, dim=2)
        return pos
    elif network_drawing_layout == "kamada_kawai_layout":
        if is_3d:
            pos = nx.kamada_kawai_layout(G, pos=coords, weight=None, dim=dim)
        else:
            pos = nx.kamada_kawai_layout(G, pos=coords, weight=None, dim=2)
        return pos
    return None  # Return None when layout is "none"

def get_coordinates(pos):
    coords = []
    for key, value in pos.items():
        coords.append([value[0], value[1], value[2]])
    coords = np.array(coords, dtype="float32")
    return coords

def get_lines_data(G, coords):
    lines = []
    for u, v in G.edges:
        # print(u, v)
        lines.append(np.array([coords[u], coords[v]], dtype="float32")) #Corrected to work with the coordinates that have been changed to be indexes not IDs
    return lines

def create_3d_plot(_G, coords, lines, edge_weights):
    plotter = pv.Plotter(window_size=[600, 600])

    task4.text("Drawing spheres...")
    pdata = pv.PolyData(coords)
    pdata['orig_sphere'] = np.arange(len(_G.nodes))

    # create many spheres from the point cloud
    # plotter = pyvista.Plotter(window_size=[600, 600])
    sphere = pv.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    # plotter.add_mesh(sphere)
    plotter.add_mesh(pc, cmap="viridis")

    task5.text("Drawing lines...")
    for i in range(len(_G.edges)):
        spline = pv.Spline(lines[i], 2).tube(radius=edge_weights[i] / (max(edge_weights) * 100))
        plotter.add_mesh(spline, color="red")

    task6.text("Sending...")
    plotter.camera.zoom(0.6)
    # Send to streamlit
    stpyvista(plotter)
    task6.text("Sent")

def create_2d_plot(A, coordinates, edges):
    fig, ax = plt.subplots(figsize=(8, 6))
    max_weight = A.data.max()
    for edge in edges:
        start, end, weight = edge["source"], edge["target"], edge["weight"]
        start_coords = coordinates[start]
        end_coords = coordinates[end]
        ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], '-', color="black", linewidth=weight*10/max_weight)
    ax.scatter(x=coordinates[:, 0], y=coordinates[:, 1], s=50, c=range(coordinates.shape[0]), cmap="viridis")
    ax.set_title("Fungi Network")
    #ax.axis("off") # can add back in as the numbers are still fairly relevant
    st.pyplot(fig)

def create_2d_plot_layout(G, pos, edge_weights):
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=list(G.nodes), cmap="viridis", ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, ax=ax)
    ax.set_title("Fungi Network")
    st.pyplot(fig)


def draw_selection(mat_selection, network_drawing_layout, is_3D):
    task1.text("Loading data...")
    A, nodes, edges, coordinates = load_mat_data(mat_selection)
    coords_3D = get_original_coordinates(coordinates, is_3d=True)
    coords_2D = get_original_coordinates(coordinates, is_3d=False)

    task2.text("Drawing network layout...")
    G = construct_graph(nodes, edges)
    edge_weights = get_edge_weights(G)

    edge_weights_G2 = [G[u][v]["weight"]/10000 for u, v in G.edges]
    max_edge_weight_G2 = max(edge_weights_G2)
    edge_weights = [weight*10/max_edge_weight_G2 for weight in edge_weights_G2]

    if network_drawing_layout != "none":
        task3.text("Processing...")
        if is_3D:
            pos = get_node_positions(network_drawing_layout, G, coords_3D, is_3d=True)
            coords = get_coordinates(pos)
            lines = get_lines_data(G, coords) #Corrected this to work with the new indexed coordinates
            create_3d_plot(G, coords, lines, edge_weights)
        else:
            pos = get_node_positions(network_drawing_layout, G, coords_2D, is_3d=False)
            create_2d_plot_layout(G, pos, edge_weights)

    else:
        task3.text("Processing...")
        if is_3D:
            st.write("Drawing 3D plot without network layout is not supported yet.")
        else:
            create_2d_plot(A, coordinates, edges.to_dict("records"))


    task1.empty()
    task2.empty()
    task3.empty()
    task4.empty()
    task5.empty()
    task6.empty()

with st.sidebar:
    mat_filenames = get_mat_filenames()
    mat_selection = st.sidebar.selectbox(
        "Please select a file",
        (mat_filenames))

    dimension = st.radio(
        "Select Dimension:",
        ("2D", "3D"))

    is_3D = dimension == "3D"


    network_drawing_layout = st.sidebar.selectbox(
        "Please select a network drawing layout",
        (NETWORK_DRAWING_LAYOUTS), id=2)

    generate_layout_button = st.button("Generate layout")

st.title("Plotting Fungal Network Data")
st.markdown("Data from: A Spatial Database of Fungi, A. Banerjee et al. Available at: [https://www.cs.cornell.edu/~arb/data/spatial-fungi/(https://www.cs.cornell.edu/~arb/data/spatial-fungi/)")

# Create placeholders for each task
task1 = st.empty()
task2 = st.empty()
task3 = st.empty()
task4 = st.empty()
task5 = st.empty()
task6 = st.empty()

if generate_layout_button:
    draw_selection(mat_selection, network_drawing_layout, is_3D)

with st.expander("Additional Citation"):
    st.markdown("""
    @article{Lee-2016-CP,
    doi = {10.1093/comnet/cnv034},
    url = {https://doi.org/10.1093/comnet/cnv034},
    year  = {2016},
    month = {apr},
    publisher = {Oxford University Press ({OUP})},
    author = {Sang Hoon Lee and Mark D. Fricker and Mason A. Porter},
    title = {Mesoscale analyses of fungal networks as an approach for quantifying phenotypic traits},
    journal = {Journal of Complex Networks}
    }
    """)