"""Isaac/Universal Scene Description adapter; import only after Kit starts."""

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from roomgraph.furnishings import MATERIALS
from roomgraph.meshes import cylinder_mesh, load_gltf, rounded_box


class SceneBuilder:
    """Author reusable materials, furniture meshes, and imported CC0 assets."""

    def __init__(self, stage, asset_root):
        self.stage = stage
        self.asset_root = Path(asset_root).resolve()
        self.materials = {}
        self.mesh_cache = {}
        self.asset_cache = {}
        self.texture_inputs = []
        for name, m in MATERIALS.items():
            fabric = name in {"sage", "ivory", "rust", "linen", "teal", "navy", "mustard"}
            self.material(
                name,
                m.color,
                m.roughness,
                m.metallic,
                m.emission,
                texture=None,
                diffuse_texture=False,
            )
            if fabric:
                self.material(
                    "fabric_" + name,
                    m.color,
                    m.roughness,
                    m.metallic,
                    m.emission,
                    texture="denim_fabric",
                    diffuse_texture=False,
                )

    def texture(self, path, name, filename, colorspace="raw", normal=False):
        shader = UsdShade.Shader.Define(self.stage, path + "/" + name)
        shader.CreateIdAttr("UsdUVTexture")
        attr = shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
        attr.Set(str(Path(filename).resolve()))
        self.texture_inputs.append(attr)
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        reader = UsdShade.Shader.Define(self.stage, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        if normal:
            shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            shader.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        return shader

    def material(
        self,
        name,
        color,
        roughness=0.6,
        metallic=0.0,
        emission=(0, 0, 0),
        texture=None,
        diffuse_texture=True,
    ):
        path = "/World/Materials/" + name
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
        if texture:
            directory = self.asset_root / texture
            normal = self.texture(path, "Normal", directory / "nor_gl.jpg", normal=True)
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                normal.ConnectableAPI(), "rgb"
            )
            if diffuse_texture:
                diffuse = self.texture(path, "Diffuse", directory / "Diffuse.jpg", "sRGB")
                shader.GetInput("diffuseColor").ConnectToSource(diffuse.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.materials[name] = material
        return material

    def mesh(self, path, positions, normals, uv, indices, face_size=4):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(positions.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([face_size] * (len(indices) // face_size))
        mesh.CreateFaceVertexIndicesAttr(indices.astype(np.int32).tolist())
        mesh.CreateNormalsAttr(normals.astype(np.float32).tolist())
        mesh.SetNormalsInterpolation("vertex")
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
        ).Set(uv.astype(np.float32).tolist())
        return mesh

    def part(self, p, root="/World/Furnishings"):
        path = root + "/" + p.name
        if p.shape == "rounded_box":
            key = (tuple(p.size), p.radius)
            if key not in self.mesh_cache:
                self.mesh_cache[key] = rounded_box(p.size, p.radius)
            mesh = self.mesh(path, *self.mesh_cache[key])
        elif p.shape == "sphere":
            mesh = UsdGeom.Sphere.Define(self.stage, path)
            mesh.CreateRadiusAttr(1)
        elif p.shape == "cylinder":
            key = ("cylinder", tuple(p.size))
            if key not in self.mesh_cache:
                self.mesh_cache[key] = cylinder_mesh(p.size)
            mesh = self.mesh(path, *self.mesh_cache[key], face_size=3)
        else:
            raise ValueError(p.shape)
        mesh.AddTranslateOp().Set(Gf.Vec3d(*p.center))
        mesh.AddRotateXYZOp().Set(Gf.Vec3f(*p.rotation))
        if p.shape == "sphere":
            mesh.AddScaleOp().Set(Gf.Vec3f(*(np.array(p.size) / 2)))
        material = p.material
        cloth = p.category in {"bed", "sofa", "chair", "rug", "curtain"} or p.name.endswith(
            "/shade"
        )
        if cloth and "fabric_" + material in self.materials:
            material = "fabric_" + material
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[material])
        return mesh

    def imported(self, asset, name, origin, extent, yaw=0, dimension="height"):
        """Fit an imported asset by height or width while preserving proportions."""
        if asset not in self.asset_cache:
            self.asset_cache[asset] = load_gltf(self.asset_root / asset / (asset + ".gltf"))
        meshes, data = self.asset_cache[asset]
        points = np.concatenate([m["positions"] for m in meshes])
        lower, upper = points.min(axis=0), points.max(axis=0)
        axis = 2 if dimension == "height" else 0
        scale = extent / (upper[axis] - lower[axis])
        center = np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]])
        parent_path = "/World/Furnishings/" + name
        parent = UsdGeom.Xform.Define(self.stage, parent_path)
        parent.AddTranslateOp().Set(Gf.Vec3d(*origin))
        parent.AddRotateZOp().Set(yaw)
        directory = self.asset_root / asset

        def image_file(texture_info):
            index = data["textures"][texture_info["index"]]["source"]
            return directory / data["images"][index]["uri"]

        for index, source in enumerate(meshes):
            mat_name = f"asset_{asset}_{source['material']}"
            if mat_name not in self.materials:
                spec = data["materials"][source["material"]]
                pbr = spec.get("pbrMetallicRoughness", {})
                factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
                material = self.material(
                    mat_name,
                    factor[:3],
                    pbr.get("roughnessFactor", 1),
                    pbr.get("metallicFactor", 1),
                )
                mat_path = str(material.GetPath())
                surface = UsdShade.Shader(self.stage.GetPrimAtPath(mat_path + "/Surface"))
                for key, input_name, channel, space in [
                    ("baseColorTexture", "diffuseColor", "rgb", "sRGB"),
                    ("metallicRoughnessTexture", "roughness", "g", "raw"),
                ]:
                    if key in pbr:
                        tex = self.texture(mat_path, key, image_file(pbr[key]), space)
                        surface.GetInput(input_name).ConnectToSource(tex.ConnectableAPI(), channel)
                        if key == "metallicRoughnessTexture":
                            # glTF packs roughness in G and metalness in B.
                            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                                Gf.Vec4f(
                                    1,
                                    pbr.get("roughnessFactor", 1),
                                    pbr.get("metallicFactor", 1),
                                    1,
                                )
                            )
                            surface.GetInput("metallic").ConnectToSource(tex.ConnectableAPI(), "b")
                if "normalTexture" in spec:
                    tex = self.texture(
                        mat_path, "Normal", image_file(spec["normalTexture"]), normal=True
                    )
                    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                        tex.ConnectableAPI(), "rgb"
                    )
            mesh = self.mesh(
                parent_path + f"/mesh_{index}",
                (source["positions"] - center) * scale,
                source["normals"],
                source["uv"],
                source["indices"],
                face_size=3,
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[mat_name])
        return {
            "asset": asset,
            "object_id": name,
            "origin": list(origin),
            "yaw_deg": yaw,
            "dimensions_m": ((upper - lower) * scale).tolist(),
            "license": "CC0-1.0",
        }
