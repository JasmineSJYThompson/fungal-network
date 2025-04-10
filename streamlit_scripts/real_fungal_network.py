from scipy.io import loadmat
import numpy as np
import pandas as pd
import networkx as nx

from pyvista import PolyData, Sphere, Spline, Plotter
import pyvista as pv
import streamlit as st

from stpyvista import stpyvista

import platform
import os

FOLDER = "/data/fungal_networks/"
#MAT_FILENAMES = ["Pp_M_Tokyo_U_N_26h_1.mat", "Pp_M_Tokyo_U_N_26h_2.mat"]
NETWORK_DRAWING_LAYOUTS = ["spring_layout", "kamada_kawai_layout"]

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

    edges = pd.DataFrame({"source": A.row, "target": A.col, "weight": A.data})
    edges = pd.read_csv(os.getcwd() + "/data/sample-network.csv")
    # Filters out very tiny weights for the purpose of a better visualisation
    edges = edges[edges["weight"] >= 1]
    # Changes to the correct format to easily add to networkx
    edges = edges.to_dict("records")

    mu = np.array(coordinates).mean()
    sigma = np.array(coordinates).std()
    coords_3D = pd.DataFrame(coordinates).to_dict("index")
    coords_3D = {key: [value[0], value[1], np.random.normal(mu, sigma)] for key, value in coords_3D.items()}

    return edges, coords_3D

def construct_graph(edges):
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    return G

def get_edge_weights(G):
    edge_weights = [G[u][v]["weight"] * 1.5 for u, v in G.edges]
    return edge_weights

def get_node_positions(network_drawing_layout, G, coords_3D):
    if network_drawing_layout == "spring_layout":
        pos = nx.spring_layout(G, seed=42, pos=coords_3D, weight=None, dim=3)
        return pos
    else:
        pos = nx.kamada_kawai_layout(G, pos=coords_3D, weight=None, dim=3)
        return pos
    return -1

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
        lines.append(np.array([coords[u - 1], coords[v - 1]], dtype="float32"))
    return lines

def create_plot(_G, coords, lines, edge_weights):
    plotter = Plotter(window_size=[600, 600])

    task4.text("Drawing spheres...")
    pdata = PolyData(coords)
    pdata['orig_sphere'] = np.arange(len(_G.nodes))

    # create many spheres from the point cloud
    # plotter = pyvista.Plotter(window_size=[600, 600])
    sphere = Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    # plotter.add_mesh(sphere)
    plotter.add_mesh(pc, cmap="Reds")

    task5.text("Drawing lines...")
    for i in range(len(_G.edges)):
        spline = Spline(lines[i], 2).tube(radius=edge_weights[i] / (max(edge_weights) * 100))
        plotter.add_mesh(spline, color="red")

    task6.text("Sending...")
    plotter.camera.zoom(0.6)
    # Send to streamlit
    stpyvista(plotter)
    task6.text("Sent")

def draw_selection(mat_selection, network_drawing_layout):
    task1.text("Loading data...")
    edges, coords_3D = load_mat_data(mat_selection)

    task2.text("Drawing network layout...")
    G = construct_graph(edges)
    edge_weights = get_edge_weights(G)
    pos = get_node_positions(network_drawing_layout, G, coords_3D)

    task3.text("Processing...")
    coords = get_coordinates(pos)
    lines = get_lines_data(G, coords)

    create_plot(G, coords, lines, edge_weights)

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
    network_drawing_layout = st.sidebar.selectbox(
        "Please select a network drawing layout",
        (NETWORK_DRAWING_LAYOUTS))

    generate_layout_button = st.button("Generate layout")

st.title("Plotting Fungal Network Data in 3D")

# Create placeholders for each task
task1 = st.empty()
task2 = st.empty()
task3 = st.empty()
task4 = st.empty()
task5 = st.empty()
task6 = st.empty()

if generate_layout_button:
    draw_selection(mat_selection, network_drawing_layout)
