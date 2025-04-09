#import os

#import time

from scipy.io import loadmat

import numpy as np
import pandas as pd

import networkx as nx

from  pyvista import PolyData, Sphere, Spline, Plotter
import streamlit as st

from stpyvista import stpyvista

folder = "../data/fungal_networks/"

#mat_filenames = os.listdir(folder)
#mat_filenames = [filename for filename in mat_filenames if filename[0:1] == "Pp"]

mat_filenames = ["Pp_M_Tokyo_U_N_26h_1.mat", "Pp_M_Tokyo_U_N_26h_2.mat"]

with st.sidebar:
    mat_selection = st.sidebar.selectbox(
        "Please select a .mat file",
        (mat_filenames))

    #with st.spinner("Wait for it...", show_time=True):
    #    time.sleep(5)
    #st.success("Done!")
    #st.button("Rerun")

import os
print(os.getwd())

data = loadmat(f"{folder}{mat_selection}")

A = data["A"].tocoo()

edges = pd.DataFrame({"source": A.row, "target": A.col, "weight": A.data})

#edges = pd.read_csv("./../data/sample-network.csv")
# Filters out very tiny weights for the purpose of a better visualisation
edges = edges[edges["weight"] >= 1]
# Changes to the correct format to easily add to networkx
edges = edges.to_dict("records")

G = nx.Graph()
for edge in edges:
    G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

edge_weights = [G[u][v]["weight"] * 1.5 for u, v in G.edges]

pos = nx.spring_layout(G, seed=42, weight=None, dim=3)

coords = []
for key, value in pos.items():
    coords.append([value[0], value[1], value[2]])
#coords = np.array(coords, dtype="float32")

lines = []
for u, v in G.edges:
    #print(u, v)
    lines.append(np.array([coords[u-1], coords[v-1]], dtype="float32"))

coords = np.array(coords, dtype="float32")

#pyvista.global_theme.allow_empty_mesh = True

plotter = Plotter(window_size=[600, 600])

for i in range(len(G.edges)):
    spline = Spline(lines[i], 2).tube(radius=edge_weights[i]/(max(edge_weights)*100))
    plotter.add_mesh(spline, color="red")

#print("Max coord:", coords.max())
#print("Min Coord:", coords.min())

pdata = PolyData(coords)
pdata['orig_sphere'] = np.arange(len(G.nodes))

# create many spheres from the point cloud
#plotter = pyvista.Plotter(window_size=[600, 600])
sphere = Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
pc = pdata.glyph(scale=False, geom=sphere, orient=False)
plotter.add_mesh(sphere)
plotter.add_mesh(pc, cmap="Reds")

spline = Spline(np.array([[1, 1, 1], [-1, -1, -1]]), 2).tube(radius=0.01)
#st.draw(spline.plot(scalars='arc_length', show_scalar_bar=False))

#plotter.add_mesh(spline, cmap="Reds")

plotter.camera.zoom(0.6)

# Send to streamlit
stpyvista(plotter)
