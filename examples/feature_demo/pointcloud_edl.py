"""
Point cloud with Eye-Dome Lighting (EDL)
=======================================

This example renders the Stanford Bunny point cloud and applies EDL as a
post-process to enhance depth perception.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import requests
import tarfile
from pathlib import Path
import os
from typing import Union
import trimesh

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pygfx.renderers.wgpu.engine.edl_effectpass import EDLEffectPass


def load_bunny(directory: Union[Path, os.PathLike, str, bytes] = "/tmp/bunny_data",
               chunk_size: int = 8192) -> trimesh.Trimesh:
    """
    Downloads the stanford bunny into the specified directory and returns the PointCloud

    Args:
        directory: The directory to load.
    Returns:
        The PointCloud containing the stanford bunny data.
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    bunny_path = directory / "bunny.tar.gz"

    # download the file if it does not exist
    if not bunny_path.exists():
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        with open(bunny_path, 'wb') as f:

            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)

    # Extract the data
    bunny_tar_file = tarfile.open(bunny_path)

    data_dir = bunny_path.parent / "bunny_data"
    data_dir.mkdir(exist_ok=True)
    bunny_tar_file.extractall(data_dir)
    bunny_tar_file.close()

    # Load the ply file from the data path
    mesh = trimesh.load_mesh(data_dir / "bunny" / "reconstruction" / "bun_zipper.ply", process=False)
    return mesh


def make_point_cloud(num_points=200_000, radius=3.0, branches=4):
    rng = np.random.default_rng(0)
    theta = rng.random(num_points).astype(np.float32) * 2 * np.pi
    r = (rng.random(num_points).astype(np.float32) ** 0.5) * radius
    z = (rng.standard_normal(num_points).astype(np.float32)) * (radius * 0.1)
    # Create spiral-ish branches for visual depth cues
    branch = rng.integers(0, branches, size=num_points)
    theta = theta + (2 * np.pi / branches) * branch
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.column_stack([x, y, z]).astype(np.float32)
    # Colors by branch
    palette = np.array(
        [
            [0.9, 0.2, 0.2, 1.0],
            [0.2, 0.9, 0.2, 1.0],
            [0.2, 0.6, 0.9, 1.0],
            [0.9, 0.8, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    colors = palette[branch % len(palette)].astype(np.float32)
    sizes = (rng.random(num_points).astype(np.float32) * 2.0 + 1.5).astype(np.float32)
    return positions, colors, sizes


canvas = RenderCanvas(update_mode="continuous")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Background for contrast
scene.add(gfx.Background.from_color("#111"))

# Load Stanford Bunny and render as point cloud (ignore faces)
mesh = load_bunny()
positions = np.asarray(mesh.vertices, dtype=np.float32)
geometry = gfx.Geometry(positions=positions)
material = gfx.PointsMaterial(size=6.0, color="#ddd", aa=True, size_space="screen")
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.PerspectiveCamera(60, 1)
camera.show_object(scene, view_dir=(1, -1, 0.8))
camera.position = (8, -8, 6)

controller = gfx.OrbitController(camera, register_events=renderer)

# Add lights (not strictly needed for points, but harmless)
scene.add(gfx.AmbientLight(0.4), camera.add(gfx.DirectionalLight(0.8)))

# Apply EDL as post-processing
renderer.effect_passes = [EDLEffectPass(strength=10.0, radius=1.5, depth_edge_threshold=0.0)]


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()


