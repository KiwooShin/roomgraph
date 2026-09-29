# Furnishing scene descriptions

JSON is the editable source of truth for a scene. Keeping room structure,
furnishings, cameras, lighting, and render settings together makes variants easy
to review in Git and reproduce. Start with `configs/scenes/bedroom.json`.

Coordinates use metres, Z up, and the room centre at floor level as the origin.
Yaw is in degrees around +Z. The current shell is a fixed 6 × 5 × 3 m room with
a north doorway and an east window opening. Its template and dimensions are
validated; arbitrary dimensions and connected rooms are not implemented yet.

`furnishings` contains named recipes with explicit parameters. For example:

```json
{
  "recipe": "workstation",
  "parameters": {
    "name": "study_desk",
    "xy": [-1.5, -1.9],
    "yaw": 0
  }
}
```

Recipes define furniture assemblies such as beds, desks, chairs, cabinets,
curtains, rugs, lamps, artwork, and plants. Their accepted parameters are the
corresponding `Furnisher` method signatures in `src/roomgraph/furnishings.py`.
The `kitchen_suite` recipe currently expands a fixed cabinet/dining layout.
The seed controls procedural detail. Asset overrides replace selected procedural
objects with textured models; the asset catalog records provenance and checksums.

Each camera has an ID, position, target, and focal length. The top-down triangle
points along the normalized XY projection of target minus position. Labels C1–C8
match the perspective views. The triangle indicates direction, not an exact
field-of-view footprint. The ceiling is hidden only for the top-down image.

The renderer writes these local outputs per scene:

- `scene.json`: the editable input used for the capture.
- `manifest.json`: input configuration and checksum, recipe geometry, imported
  asset placements, structural edges, camera calibration, and output metadata.
- `room.usda`: the authored scene, with relative references to the local asset cache.
- Perspective RGB, visible/hidden/full edge masks, overlays, depth and normals.
- Top-down images with and without camera markers.

In the manifest, `parts` is the procedural recipe expansion before overrides.
Entries in `imported_objects` replace recipe parts with the same object ID; they
record the asset, placement, rotation, and fitted dimensions. `part_count` counts
recipe parts, not the final imported meshes. Use the USD scene for the final
authored geometry and materials. Keep the downloaded assets alongside the project
when reopening the USD scene.

This separation gives us a practical foundation for future dataset generation:
version-controlled inputs, explicit assets, and capture manifests. Later versions
can add building layouts, collision/accessibility checks, asset variation, and
train/validation/test splits by building rather than by image.
