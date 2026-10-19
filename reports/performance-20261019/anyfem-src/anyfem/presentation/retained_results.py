"""Retained display batches for the application's existing result scenes.

Nodal fields retain the historical, explicit element-mean display reduction.
The source Field remains unchanged for probes and export.
"""
import hashlib
import numpy as np


def _digest_value(digest, value):
    """Exact scene geometry/style bytes; never NumPy's abbreviated repr."""
    from dataclasses import fields, is_dataclass
    import struct
    if isinstance(value, np.ndarray):
        digest.update(b'array'+value.dtype.str.encode()+str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif is_dataclass(value):
        digest.update(type(value).__qualname__.encode())
        for item in fields(value):
            if item.name in {'legend','result_field','result_mesh','result_colormap','result_scalar_range'}:continue
            digest.update(item.name.encode());_digest_value(digest,getattr(value,item.name))
    elif isinstance(value, dict):
        for key in sorted(value):_digest_value(digest,key);_digest_value(digest,value[key])
    elif isinstance(value, (tuple,list)):
        digest.update(b"sequence:"+str(len(value)).encode()+b"[")
        for item in value:_digest_value(digest,item)
        digest.update(b"]")
    elif isinstance(value,(float,np.floating)):digest.update(struct.pack('!d',float(value)))
    elif isinstance(value,(str,bytes,int,np.integer,bool,type(None))):digest.update(type(value).__name__.encode()+b":"+repr(value).encode()+b";")
    else:raise TypeError(f'unsupported retained scene identity: {type(value).__name__}')


def element_id(owners):
    return next((int(owner.id) for owner in owners if getattr(owner,"kind",None)=="element"), None)


def prepare(viewport, scene):
    from any3dview import MeshArrays, PackedOwnerTable
    field, source_mesh = scene.result_field, scene.result_mesh
    if field is None or source_mesh is None:
        return None
    def scalar(eid):
        if field.per_element:
            return field.element_values.get(eid,np.nan)
        nodes = source_mesh.shells[eid] if eid in source_mesh.shells else source_mesh.beams.get(eid,())
        values = [field.node_values.get(node,np.nan) for node in nodes]
        return float(np.mean(values)) if values and np.isfinite(values).all() else np.nan
    groups = {}
    remaining = scene.copy()
    remaining.result_field = remaining.result_mesh = None
    remaining.faces, remaining.lines = [], []
    for patch in scene.faces:
        if not patch.polygons or any(element_id(patch.owners_for_polygon(i)) is None for i in range(len(patch.polygons))):
            remaining.faces.append(patch);continue
        style = viewport._face_style(patch)
        group = groups.setdefault(("face",)+style, [])
        for index, polygon in enumerate(patch.polygons):
            owners = patch.owners_for_polygon(index)
            group.append((np.asarray(polygon), element_id(owners), viewport._pick_binding(owners,patch.tag)))
    for line in scene.lines:
        eid = element_id(line.pick_owners)
        if eid is None or (not viewport.visualization.show_model_lines and not line.annotation):
            remaining.lines.append(line);continue
        group = groups.setdefault(("line",int(line.width),bool(line.draw_overlay)), [])
        binding = viewport._pick_binding(line.pick_owners,line.tag)
        points = np.asarray(line.points)
        for start,end in zip(points,points[1:]):group.append((np.asarray([start,end]),eid,binding))
    batches = []
    digest = hashlib.sha256()
    _digest_value(digest,viewport.visualization)
    for key, primitives in groups.items():
        positions, indices, mappings, bindings = [], [], [], []
        ids = tuple(dict.fromkeys(eid for points,eid,binding in primitives))
        dense = {eid:i for i,eid in enumerate(ids)}
        for points,eid,binding in primitives:
            offset = len(positions);positions.extend(points)
            if key[0]=="face":
                for index in range(1,len(points)-1):
                    indices.append((offset,offset+index,offset+index+1));mappings.append(dense[eid]);bindings.append(binding)
            else:indices.append((offset,offset+1));mappings.append(dense[eid]);bindings.append(binding)
        positions = np.asarray(positions,dtype=np.float64).reshape(-1,3)
        indices = np.asarray(indices,dtype=np.uint32).reshape(-1,3 if key[0]=="face" else 2)
        mappings = np.asarray(mappings,dtype=np.uint32)
        values = np.asarray([scalar(eid) for eid in ids],dtype=np.float32)
        _digest_value(digest,(key,ids,bindings));digest.update(positions.tobytes());digest.update(indices.tobytes())
        options = dict(owners=PackedOwnerTable.from_owners(**{ "triangles" if key[0]=="face" else "lines":bindings}),scalar_range=scene.result_scalar_range)
        options["colormap"] = scene.result_colormap
        if key[0]=="face":
            # Shells represent surfaces, not closed solids. Their selected
            # result field must remain visible from either viewing direction.
            options.update(outline=key[1],opacity=key[2],lit=key[3],width=viewport.visualization.edge_width,cull_backface=False)
            mesh = MeshArrays(positions,indices,triangle_to_element=mappings,element_ids=np.asarray(ids,dtype=np.uint64),element_scalars=values)
        else:
            options.update(line_width=key[1],draw_overlay=key[2],layer=30)
            mesh = MeshArrays(positions,np.empty((0,3),dtype=np.uint32),lines=indices,line_to_element=mappings,element_ids=np.asarray(ids,dtype=np.uint64),element_scalars=values)
        batches.append((mesh,options))
    # Overlay/style changes require a redraw; component legends do not.
    saved_legend = remaining.legend;remaining.legend=None
    _digest_value(digest,remaining);remaining.legend=saved_legend
    return digest.digest(),batches,remaining


def scalar_values(field, source_mesh, ids):
    if field.per_element:
        return np.asarray([field.element_values.get(int(eid),np.nan) for eid in ids],dtype=np.float32)
    rows=[]
    for eid in ids:
        eid=int(eid)
        nodes=source_mesh.shells[eid] if eid in source_mesh.shells else source_mesh.beams.get(eid,())
        values=[field.node_values.get(node,np.nan) for node in nodes]
        rows.append(float(np.mean(values)) if values and np.isfinite(values).all() else np.nan)
    return np.asarray(rows,dtype=np.float32)


def prepare_warm_field(field, scene, retained, *, limits=None, display_units='SI (m / Pa)', colormap=None, working_bytes=128*1024*1024):
    """Worker-side scalar conversion and software fallback colours."""
    import sys
    from anypost.errors import WorkingSetError
    from ..ui.scene import result_display_metadata, _ramp_color, COLOR_MESH_FILL, COLOR_BEAM
    count=sum(len(patch.polygons) for patch in scene.faces)+len(scene.lines)
    # Scalar outputs, temporary mean rows and retained colour strings/lists.
    estimate=count*128+sum(len(ids)*32 for ids in retained[2])+16384
    if estimate>working_bytes:raise WorkingSetError('retained display conversion exceeds working cap')
    scalars=tuple(scalar_values(field,retained[3],ids) for ids in retained[2])
    range_,palette,legend=result_display_metadata(field,limits=limits,display_units=display_units,colormap=colormap)
    low,high=range_
    def color(eid,fallback):
        if field.per_element:value=field.element_values.get(eid,np.nan)
        else:
            mesh=scene.result_mesh
            nodes=mesh.shells[eid] if eid in mesh.shells else mesh.beams.get(eid,())
            samples=[field.node_values.get(node,np.nan) for node in nodes]
            value=float(np.mean(samples)) if samples and np.isfinite(samples).all() else np.nan
        return _ramp_color((float(value)-low)/(high-low),palette) if np.isfinite(value) else fallback
    faces=[];lines=[]
    for patch in scene.faces:
        colors=list(patch.colors)
        for i in range(len(patch.polygons)):
            eid=element_id(patch.owners_for_polygon(i))
            if eid is not None:colors[i]=color(eid,COLOR_MESH_FILL)
        faces.append((len(faces),colors))
    for index,line in enumerate(scene.lines):
        eid=element_id(line.pick_owners)
        if eid is not None:lines.append((index,color(eid,COLOR_BEAM)))
    colors=(faces,lines)
    resident=sum(a.nbytes for a in scalars)+sum(sys.getsizeof(c)+sum(sys.getsizeof(v) for v in c) for p,c in faces)+sum(sys.getsizeof(c) for p,c in lines)
    field._display_buffers=(retained[0],scalars,colors)
    field._display_buffer_bytes=resident


def cpu_color_bytes(scene,updates=None):
    import sys
    seen=set();total=0
    def count(value):
        nonlocal total
        if id(value) not in seen:seen.add(id(value));total+=sys.getsizeof(value)
    faces,lines=updates if updates is not None else (tuple((i,p.colors) for i,p in enumerate(scene.faces)),tuple((i,p.color) for i,p in enumerate(scene.lines)))
    for i,colors in faces:
        count(colors)
        for color in colors:count(color)
    for i,color in lines:count(color)
    return total


def prepared_publication_bytes(field,scene,scalar_bytes):
    prepared=getattr(field,'_display_buffers',None)
    return int(scalar_bytes)+cpu_color_bytes(scene,None if prepared is None else prepared[2])


def discard_prepared_field(field):
    field.__dict__.pop('_display_buffers',None);field.__dict__.pop('_display_buffer_bytes',None)
