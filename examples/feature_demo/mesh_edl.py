"""
Stanford Bunny mesh with Eye-Dome Lighting (EDL)
================================================

This example loads the Stanford Bunny as a triangle mesh, renders it using a
Phong material, and applies EDL as a post-process to enhance shape perception.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import requests
import tarfile
from pathlib import Path
import os
from typing import Union
import trimesh

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pygfx.renderers.wgpu.engine.edl_effectpass import EDLEffectPass


def load_bunny(
    directory: Union[Path, os.PathLike, str, bytes] = "/tmp/bunny_data",
    chunk_size: int = 8192,
) -> trimesh.Trimesh:
    """
    Downloads the stanford bunny into the specified directory and returns the mesh.
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    bunny_path = directory / "bunny.tar.gz"

    if not bunny_path.exists():
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        with open(bunny_path, "wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get("content-length")
            if total_length is None:
                f.write(response.content)
            else:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)

    bunny_tar_file = tarfile.open(bunny_path)
    data_dir = bunny_path.parent / "bunny_data"
    data_dir.mkdir(exist_ok=True)
    bunny_tar_file.extractall(data_dir)
    bunny_tar_file.close()

    mesh = trimesh.load_mesh(
        data_dir / "bunny" / "reconstruction" / "bun_zipper.ply", process=False
    )
    return mesh


canvas = RenderCanvas(update_mode="continuous")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#111"))

tm = load_bunny()
geometry = gfx.geometry_from_trimesh(tm)
material = gfx.MeshBasicMaterial(color="#e6e6e6")
mesh = gfx.Mesh(geometry, material)
scene.add(mesh)

camera = gfx.PerspectiveCamera(50, 1)
camera.show_object(mesh, view_dir=(1, -1, 0.6))
camera.position = (0.02, -0.04, 0.015)

controller = gfx.OrbitController(camera, register_events=renderer)

scene.add(gfx.AmbientLight(0.2), camera.add(gfx.DirectionalLight(1.0)))

# Apply EDL post-processing (Potree-like scaling inside pass)
renderer.effect_passes = [
    EDLEffectPass(strength=10.0, radius=1.5, depth_edge_threshold=0.0)
]


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
