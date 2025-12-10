bl_info = {
    "name": "Atlas Lightmap Baker",
    "author": "SunJay",
    "version": (1, 10, 2),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Lightmap",
    "description": "Bake atlas lightmaps for user-defined groups of objects, with preview toggle.",
    "category": "Render",
}

import bpy
import random
import math
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import IntProperty, PointerProperty


# ------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------

class AtlasLightmapSettings(PropertyGroup):
    image_size: IntProperty(
        name="Image Size",
        description="Resolution of the lightmap atlases (width=height)",
        default=2048,
        min=128,
        max=8192,
    )

    margin: IntProperty(
        name="Margin (px)",
        description="Bake margin in pixels",
        default=16,
        min=0,
        max=64,
    )

    samples: IntProperty(
        name="Samples",
        description="Cycles samples used for baking",
        default=512,
        min=1,
    )

    group_size: IntProperty(
        name="Objects per Lightmap",
        description="How many objects to pack into each single lightmap atlas",
        default=5,
        min=1,
    )


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def get_bake_objects(context):
    sel_meshes = [o for o in context.selected_objects if o.type == 'MESH']
    if sel_meshes:
        return sel_meshes
    return [o for o in context.scene.objects if o.type == 'MESH']

def ensure_lightmap_uv(obj):
    if obj.type != 'MESH' or not obj.data:
        return
    mesh = obj.data
    uv_layers = mesh.uv_layers
    if len(uv_layers) == 0:
        uv_layers.new(name="UVMap")
    lm_layer = uv_layers.get("Lightmap")
    if lm_layer is None:
        uv_layers.new(name="Lightmap")

def make_materials_single_user(objs):
    """Ensure each mesh object in `objs` has unique material instances.

    This prevents a single Material datablock from being used in more than one
    lightmap group, which would otherwise cause Lightmap image nodes to be
    overwritten as groups are baked (leading to black or incorrect lightmaps).
    We only duplicate when a material has more than one user, so running this
    repeatedly will not explode the material count.
    """
    for obj in objs:
        if obj.type != 'MESH' or not obj.data:
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            # If this material datablock is shared by multiple users, make a copy
            # just for this object. After we reassign, the original material's
            # user count drops, so subsequent objects won't keep copying it.
            if getattr(mat, "users", 1) > 1:
                new_mat = mat.copy()
                # Keep a readable name that hints it's a lightmap-specific copy
                try:
                    new_mat.name = f"{mat.name}_LM_{obj.name}"
                except Exception:
                    # Fallback if naming fails for some reason
                    pass
                slot.material = new_mat


def setup_gpu(scene):
    scene.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        compute_device_type = prefs.compute_device_type
        devices = prefs.get_devices_for_type(compute_device_type)
        any_gpu = any(d.type == 'CUDA' or d.type == 'OPTIX' or d.type == 'HIP' or d.type == 'METAL'
                      for d in devices)
        if any_gpu:
            prefs.compute_device_type = compute_device_type
            prefs.refresh_devices()
            scene.cycles.device = 'GPU'
        else:
            scene.cycles.device = 'CPU'
    except Exception:
        scene.cycles.device = 'CPU'


def setup_cycles_for_baking(scene, samples):
    c = scene.cycles

    # Use relatively high samples for cleaner results
    if hasattr(c, "samples"):
        c.samples = samples
    if hasattr(c, "preview_samples"):
        c.preview_samples = samples

    # Adaptive sampling — essential for clean denoising
    if hasattr(c, "use_adaptive_sampling"):
        c.use_adaptive_sampling = True
    if hasattr(c, "adaptive_threshold"):
        c.adaptive_threshold = 0.005  # Even tighter for less noise
    if hasattr(c, "adaptive_min_samples"):
        c.adaptive_min_samples = max(128, samples // 2)  # Higher minimum

    # Reduce bounces for speed, but keep enough for GI
    fast_bounce_settings = [
        ("max_bounces", 4),
        ("diffuse_bounces", 2),
        ("glossy_bounces", 2),
        ("transmission_bounces", 2),
        ("transparent_max_bounces", 8),
        ("volume_bounces", 0),
    ]
    for attr, val in fast_bounce_settings:
        if hasattr(c, attr):
            setattr(c, attr, val)

    # Turn off caustics to reduce fireflies
    if hasattr(c, "caustics_reflective"):
        c.caustics_reflective = False
    if hasattr(c, "caustics_refractive"):
        c.caustics_refractive = False

    # Simplify settings for faster bakes
    if hasattr(scene, "render"):
        if hasattr(scene.render, "use_simplify"):
            scene.render.use_simplify = True
        if hasattr(scene.render, "simplify_subdivision_render"):
            scene.render.simplify_subdivision_render = 0  # No subdiv during bake


def setup_bake_settings_for_diffuse_light(scene, margin_px):
    scene.render.engine = 'CYCLES'
    scene.cycles.bake_type = 'DIFFUSE'

    if hasattr(scene.render, "bake"):
        b = scene.render.bake
        if hasattr(b, "use_pass_direct"):
            b.use_pass_direct = True
        if hasattr(b, "use_pass_indirect"):
            b.use_pass_indirect = True
        if hasattr(b, "use_pass_color"):
            b.use_pass_color = False
        if hasattr(b, "use_clear"):
            b.use_clear = True
        if hasattr(b, "margin"):
            b.margin = margin_px


def find_principled_for_material(mat):
    if not mat.use_nodes or not mat.node_tree:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def make_safe_name(raw: str) -> str:
    if not raw:
        return "Unnamed"
    safe = []
    for ch in raw:
        if ch.isalnum() or ch in (" ", ".", "_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def group_objects_by_count(objs, group_size: int):
    if group_size <= 0:
        group_size = len(objs)
    objs_sorted = sorted(objs, key=lambda o: o.name)
    groups = []
    for i in range(0, len(objs_sorted), group_size):
        chunk = objs_sorted[i:i + group_size]
        if chunk:
            groups.append(chunk)
    return groups

# ------------------------------------------------------------------------
# Bake Operator (now with better noise reduction)
# ------------------------------------------------------------------------

class LIGHTMAP_OT_bake_atlas(Operator):
    bl_idname = "lightmap.bake_atlas"
    bl_label = "Bake Atlas Lightmaps"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        settings: AtlasLightmapSettings = scene.lightmap_settings

        setup_gpu(scene)
        setup_cycles_for_baking(scene, settings.samples)
        setup_bake_settings_for_diffuse_light(scene, settings.margin)

        objs = get_bake_objects(context)
        if not objs:
            self.report({'WARNING'}, "No mesh objects found to bake")
            return {'CANCELLED'}

        # Make sure each bake object has its *own* unique material instances.
        # This avoids one shared material being modified multiple times for
        # different lightmap groups, which used to cause some meshes to appear
        # completely black or sample from the wrong lightmap.
        make_materials_single_user(objs)

        # Collect all materials used by the bake objects so we can disable their normal maps
        bake_materials = set()
        for obj in objs:
            for slot in obj.material_slots:
                if slot.material:
                    bake_materials.add(slot.material)

        # Temporarily strip normal-map inputs so they are NOT baked into the lightmap
        normal_link_state = strip_normal_inputs(bake_materials)

        for obj in objs:
            ensure_lightmap_uv(obj)

        groups = group_objects_by_count(objs, settings.group_size)

        prev_active = context.view_layer.objects.active
        prev_selection = list(context.selected_objects)
        prev_mode = prev_active.mode if prev_active else 'OBJECT'

        try:
            for group_index, group in enumerate(groups):
                if not group:
                    continue

                img_name = f"Lightmap_Group_{group_index:03d}"
                image = bpy.data.images.get(img_name)
                if image is None:
                    image = bpy.data.images.new(
                        name=img_name,
                        width=settings.image_size,
                        height=settings.image_size,
                        alpha=False,
                        float_buffer=True,  # HDR for better denoising
                    )
                    image.filepath = f"//{img_name}.png"

                ensure_lightmap_nodes_for_group(group, image)

                # --- SAVE ORIGINAL NODE LINKS BEFORE BAKING ---
                material_state = {}
                materials = set()
                for obj in group:
                    for slot in obj.material_slots:
                        if slot.material:
                            materials.add(slot.material)

                for mat in materials:
                    if not mat.use_nodes or not mat.node_tree:
                        continue

                    nt = mat.node_tree
                    principled = find_principled_for_material(mat)
                    if principled is None:
                        continue

                    base_input = principled.inputs.get("Base Color")
                    if base_input is None:
                        continue

                    # Save all links
                    links_data = []
                    for link in list(base_input.links):
                        links_data.append((link.from_node.name, link.from_socket.name,
                                        link.to_node.name, link.to_socket.name))
                        nt.links.remove(link)  # temporarily disconnect
                    material_state[mat.name] = links_data

                # --- FORCE LIGHTMAP UV NODE FOR BAKING ---
                for mat in materials:
                    nt = mat.node_tree
                    nodes = nt.nodes
                    links = nt.links

                    # Find or create UVMap node for Lightmap
                    uv_node = None
                    for n in nodes:
                        if n.type == "UVMAP" and getattr(n, "uv_map", "") == "Lightmap":
                            uv_node = n
                            break
                    if uv_node is None:
                        uv_node = nodes.new("ShaderNodeUVMap")
                        uv_node.uv_map = "Lightmap"
                        uv_node.name = "LightmapUV"

                    # Connect UVMap to Lightmap image node(s)
                    for img_node in [n for n in nodes if n.type == "TEX_IMAGE" and n.image == image]:
                        if not img_node.inputs["Vector"].is_linked:
                            links.new(uv_node.outputs["UV"], img_node.inputs["Vector"])

                prev_uv_indices = {}
                for obj in group:
                    if obj.type != 'MESH' or not obj.data:
                        continue
                    uv_layers = obj.data.uv_layers
                    if not uv_layers:
                        continue
                    prev_active_idx = uv_layers.active_index
                    prev_render_idx = getattr(uv_layers, "active_render_index", prev_active_idx)
                    prev_uv_indices[obj.name] = {
                        "active_index": prev_active_idx,
                        "render_index": prev_render_idx,
                    }
                    lm_layer = uv_layers.get("Lightmap")
                    if lm_layer is not None:
                        lm_index = uv_layers.find(lm_layer.name)
                        if lm_index != -1:
                            uv_layers.active_index = lm_index
                            try:
                                if hasattr(uv_layers, "active_render_index"):
                                    uv_layers.active_render_index = lm_index
                                else:
                                    for i, layer in enumerate(uv_layers):
                                        try:
                                            layer.active_render = (i == lm_index)
                                        except AttributeError:
                                            pass
                            except Exception:
                                pass

                if context.object and context.object.mode != 'OBJECT':
                    try:
                        bpy.ops.object.mode_set(mode='OBJECT')
                    except RuntimeError:
                        pass
                for obj in context.scene.objects:
                    obj.select_set(False)
                for obj in group:
                    obj.select_set(True)
                context.view_layer.objects.active = group[0]

                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')

                if settings.image_size > 0:
                    margin_uv = float(settings.margin) / float(settings.image_size)
                else:
                    margin_uv = 0.0
                margin_uv = min(max(margin_uv, 0.0), 0.02)

                try:
                    bpy.ops.uv.lightmap_pack(
                        PREF_CONTEXT='ALL_FACES',
                        PREF_PACK_IN_ONE=True,
                        PREF_NEW_UVLAYER=False,
                        PREF_IMG_PX_SIZE=settings.image_size,
                        PREF_BOX_DIV=12,
                        PREF_MARGIN_DIV=0.1,
                    )
                except TypeError:
                    bpy.ops.uv.smart_project(
                        angle_limit=math.radians(66.0),
                        island_margin=margin_uv,
                    )
                try:
                    bpy.ops.uv.pack_islands(rotate=True, margin=margin_uv)
                except Exception:
                    pass

                bpy.ops.object.mode_set(mode='OBJECT')

                for obj in context.scene.objects:
                    obj.select_set(False)
                for obj in group:
                    obj.select_set(True)
                context.view_layer.objects.active = group[0]

                try:
                    bpy.ops.object.bake(
                        type='DIFFUSE',
                        pass_filter={'DIRECT', 'INDIRECT'},
                        margin=settings.margin,
                        margin_type='EXTEND',
                        use_clear=True,
                        target='IMAGE_TEXTURES',
                    )
                except TypeError:
                    try:
                        bpy.ops.object.bake(
                            type='DIFFUSE',
                            pass_filter={'DIRECT', 'INDIRECT'},
                            margin=settings.margin,
                            use_clear=True,
                        )
                    except TypeError:
                        bpy.ops.object.bake(
                            type='DIFFUSE',
                            margin=settings.margin,
                            use_clear=True,
                        )

                # --- RESTORE ORIGINAL MATERIAL LINKS ---
                for mat_name, links_data in material_state.items():
                    mat = bpy.data.materials.get(mat_name)
                    if not mat or not mat.node_tree:
                        continue

                    nt = mat.node_tree
                    links = nt.links

                    for from_node_name, from_socket_name, to_node_name, to_socket_name in links_data:
                        from_node = nt.nodes.get(from_node_name)
                        to_node = nt.nodes.get(to_node_name)
                        if from_node and to_node:
                            from_sock = from_node.outputs.get(from_socket_name)
                            to_sock = to_node.inputs.get(to_socket_name)
                            if from_sock and to_sock:
                                links.new(from_sock, to_sock)

                for obj in group:
                    if obj.type != 'MESH' or not obj.data:
                        continue
                    uv_layers = obj.data.uv_layers
                    if not uv_layers:
                        continue
                    prev_info = prev_uv_indices.get(obj.name)
                    if not prev_info:
                        continue
                    prev_active_idx = prev_info["active_index"]
                    prev_render_idx = prev_info["render_index"]
                    if 0 <= prev_active_idx < len(uv_layers):
                        uv_layers.active_index = prev_active_idx
                    try:
                        if hasattr(uv_layers, "active_render_index") and 0 <= prev_render_idx < len(uv_layers):
                            uv_layers.active_render_index = prev_render_idx
                        else:
                            for i, layer in enumerate(uv_layers):
                                try:
                                    layer.active_render = (i == prev_render_idx)
                                except AttributeError:
                                    pass
                    except Exception:
                        pass

        finally:
            # Restore any stripped normal-map links now that baking is done
            restore_normal_inputs(normal_link_state)
            for obj in context.scene.objects:
                obj.select_set(False)
            for obj in prev_selection:
                if obj.name in context.scene.objects:
                    obj.select_set(True)
            if prev_active and prev_active.name in context.scene.objects:
                context.view_layer.objects.active = prev_active
                try:
                    bpy.ops.object.mode_set(mode=prev_mode)
                except RuntimeError:
                    pass

        self.report({'INFO'}, f"Baking complete. If noise persists, try 512+ samples or check scene lights.")
        return {'FINISHED'}

# ------------------------------------------------------------------------
# Lightmap node setup per group
# ------------------------------------------------------------------------

def get_lightmap_node(mat):
    if mat.node_tree is None:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and (node.name == "Lightmap" or node.label == "Lightmap") and node.image:
            return node
    return None

def ensure_lightmap_nodes_for_group(objs, image):
    materials = set()
    for obj in objs:
        for slot in obj.material_slots:
            if slot.material:
                materials.add(slot.material)
    for mat in materials:
        if mat.node_tree is None:
            mat.use_nodes = True
        ntree = mat.node_tree
        nodes = ntree.nodes
        links = ntree.links
        to_remove = []
        for n in nodes:
            if n.type == 'TEX_IMAGE':
                is_named_lightmap = (n.name == "Lightmap" or n.label == "Lightmap")
                has_lightmap_image = (n.image is not None and n.image.name.startswith("Lightmap_"))
                if is_named_lightmap or has_lightmap_image:
                    to_remove.append(n)
        for n in to_remove:
            nodes.remove(n)
        lm_node = nodes.new("ShaderNodeTexImage")
        lm_node.name = "Lightmap"
        lm_node.label = "Lightmap"
        lm_node.image = image
        try:
            nodes.active = lm_node
        except Exception:
            pass
        uv_node = None
        for n in nodes:
            if n.type == 'UVMAP' and getattr(n, "uv_map", "") == "Lightmap":
                uv_node = n
                break
        if uv_node is None:
            uv_node = nodes.new("ShaderNodeUVMap")
            uv_node.uv_map = "Lightmap"
            uv_node.name = "LightmapUV"
        vec_input = lm_node.inputs.get("Vector")
        if vec_input is not None and not vec_input.is_linked:
            uv_out = uv_node.outputs.get("UV")
            if uv_out:
                links.new(uv_out, vec_input)
        principled_nodes = [n for n in nodes if n.type == 'BSDF_PRINCIPLED']
        for p in principled_nodes:
            emission_input = None
            for inp in p.inputs:
                if "Emission" in inp.name:
                    emission_input = inp
                    break
            if emission_input is None:
                continue
            while emission_input.links:
                links.remove(emission_input.links[0])
            color_out = lm_node.outputs.get("Color") or lm_node.outputs[0]
            links.new(color_out, emission_input)
            for inp in p.inputs:
                if "Emission Strength" in inp.name or inp.name == "Strength":
                    try:
                        inp.default_value = 1.0
                    except Exception:
                        pass
                    break



# ------------------------------------------------------------------------
# Normal-map stripping for bake (ignore normal maps in lightmap)
# ------------------------------------------------------------------------

def strip_normal_inputs(materials):
    """Temporarily remove links into Principled BSDF Normal inputs.

    Returns a dict so we can restore them afterwards.
    """
    state = {}

    for mat in materials:
        if not mat or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
            continue

        nt = mat.node_tree
        links = nt.links

        mat_links = []

        for node in nt.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                normal_input = node.inputs.get("Normal")
                if normal_input is None:
                    continue

                for link in list(normal_input.links):
                    mat_links.append((
                        link.from_node.name,
                        link.from_socket.name,
                        link.to_node.name,
                        link.to_socket.name,
                    ))
                    links.remove(link)

        if mat_links:
            state[mat.name] = mat_links

    return state


def restore_normal_inputs(state):
    """Restore the Principled Normal links removed by strip_normal_inputs."""
    for mat_name, links_data in state.items():
        mat = bpy.data.materials.get(mat_name)
        if not mat or mat.node_tree is None:
            continue

        nt = mat.node_tree
        links = nt.links

        for from_node_name, from_socket_name, to_node_name, to_socket_name in links_data:
            from_node = nt.nodes.get(from_node_name)
            to_node = nt.nodes.get(to_node_name)
            if from_node and to_node:
                from_sock = from_node.outputs.get(from_socket_name)
                to_sock = to_node.inputs.get(to_socket_name)
                if from_sock and to_sock:
                    links.new(from_sock, to_sock)


# ------------------------------------------------------------------------
# Preview Operators
# ------------------------------------------------------------------------

_preview_uv_state = {}
_preview_material_state = {}


class LIGHTMAP_OT_preview_enable(Operator):
    bl_idname = "lightmap.preview_enable"
    bl_label = "Preview On"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global _preview_uv_state, _preview_material_state
        _preview_uv_state = {}
        _preview_material_state = {}
        objs = get_bake_objects(context)
        mats_to_process = set()

        for obj in objs:
            if obj.type != 'MESH' or not obj.data:
                continue
            uv_layers = obj.data.uv_layers
            if not uv_layers:
                continue
            lm_layer = uv_layers.get("Lightmap")
            if lm_layer is None:
                continue
            prev_active = uv_layers.active_index
            prev_render = getattr(uv_layers, "active_render_index", prev_active)
            _preview_uv_state[obj.name] = {
                "active_index": prev_active,
                "render_index": prev_render,
            }
            lm_index = uv_layers.find(lm_layer.name)
            if lm_index != -1:
                uv_layers.active_index = lm_index
                try:
                    if hasattr(uv_layers, "active_render_index"):
                        uv_layers.active_render_index = lm_index
                    else:
                        for i, layer in enumerate(uv_layers):
                            try:
                                layer.active_render = (i == lm_index)
                            except AttributeError:
                                pass
                except Exception:
                    pass

            for slot in obj.material_slots:
                if slot.material:
                    mats_to_process.add(slot.material)

        for mat in mats_to_process:
            if not mat.use_nodes or not mat.node_tree:
                continue
            principled = find_principled_for_material(mat)
            lm_node = get_lightmap_node(mat)
            if principled is None or lm_node is None:
                continue
            nt = mat.node_tree
            base_input = principled.inputs.get("Base Color")
            if base_input is None:
                continue
            link = base_input.links[0] if base_input.is_linked else None
            _preview_material_state[mat.name] = {
                "principled_name": principled.name,
                "had_link": bool(link),
                "from_node_name": link.from_node.name if link else None,
                "from_socket_name": link.from_socket.name if link else None,
                "default_color": list(base_input.default_value),
            }
            while base_input.links:
                nt.links.remove(base_input.links[0])
            out_sock = lm_node.outputs.get("Color") or lm_node.outputs[0]
            nt.links.new(out_sock, base_input)

        self.report({'INFO'}, "Lightmap preview enabled (Lightmap UV set as render-active)")
        return {'FINISHED'}


class LIGHTMAP_OT_preview_disable(Operator):
    bl_idname = "lightmap.preview_disable"
    bl_label = "Preview Off"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global _preview_uv_state, _preview_material_state

        objs = get_bake_objects(context)
        for obj in objs:
            if obj.type != 'MESH' or not obj.data:
                continue
            uv_layers = obj.data.uv_layers
            if not uv_layers:
                continue
            prev_info = _preview_uv_state.get(obj.name)
            if not prev_info:
                continue
            prev_active = prev_info["active_index"]
            prev_render = prev_info["render_index"]
            if 0 <= prev_active < len(uv_layers):
                uv_layers.active_index = prev_active
            try:
                if hasattr(uv_layers, "active_render_index") and 0 <= prev_render < len(uv_layers):
                    uv_layers.active_render_index = prev_render
                else:
                    for i, layer in enumerate(uv_layers):
                        try:
                            layer.active_render = (i == prev_render)
                        except AttributeError:
                            pass
            except Exception:
                pass

        for mat_name, info in _preview_material_state.items():
            mat = bpy.data.materials.get(mat_name)
            if not mat or not mat.node_tree:
                continue
            nt = mat.node_tree
            nodes = nt.nodes
            principled = nodes.get(info["principled_name"])
            if principled is None:
                continue
            base_input = principled.inputs.get("Base Color")
            if base_input is None:
                continue
            while base_input.links:
                nt.links.remove(base_input.links[0])
            if info["had_link"]:
                from_node = nodes.get(info["from_node_name"]) if info["from_node_name"] else None
                if from_node:
                    from_sock = from_node.outputs.get(info["from_socket_name"]) if info["from_socket_name"] else None
                    if from_sock:
                        nt.links.new(from_sock, base_input)
            else:
                base_input.default_value = info["default_color"]

        _preview_uv_state = {}
        _preview_material_state = {}

        self.report({'INFO'}, "Lightmap preview disabled (UVs & materials restored)")
        return {'FINISHED'}

# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------

class LIGHTMAP_PT_bake_panel(Panel):
    bl_label = "Atlas Lightmap Baker"
    bl_idname = "LIGHTMAP_PT_bake_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightmap"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings: AtlasLightmapSettings = scene.lightmap_settings

        col = layout.column(align=True)
        col.label(text="Lightmap Atlases:")
        col.prop(settings, "image_size")
        col.prop(settings, "margin")
        col.prop(settings, "samples")
        col.prop(settings, "group_size")

        col.separator()
        col.operator("lightmap.bake_atlas", icon='RENDER_STILL')

        col.separator()
        row = col.row(align=True)
        row.operator("lightmap.preview_enable", icon='HIDE_OFF')
        row.operator("lightmap.preview_disable", icon='HIDE_ON')


classes = (
    AtlasLightmapSettings,
    LIGHTMAP_OT_bake_atlas,
    LIGHTMAP_OT_preview_enable,
    LIGHTMAP_OT_preview_disable,
    LIGHTMAP_PT_bake_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lightmap_settings = PointerProperty(type=AtlasLightmapSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lightmap_settings

if __name__ == "__main__":
    register()
