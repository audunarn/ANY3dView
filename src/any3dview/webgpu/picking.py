"""Conservative vector point culling with canonical projection/selection narrow phase.

No integer GPU picking claim: world/projection arrays are display-only caches.
Dense primitive ordinals count all canonically admitted primitives before culling.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from ..retained_viewer import RetainedViewer
from ..selection import ProjectedSelectionIndex


def _screen(camera,width,height,fov):
    scale=.5*height/np.tan(fov*.5)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.column_stack((width*.5+camera[:,0]*scale/camera[:,2],height*.5-camera[:,1]*scale/camera[:,2]))


def _bbox(points):
    return np.column_stack((points[:,:,0].min(axis=1),points[:,:,1].min(axis=1),points[:,:,0].max(axis=1),points[:,:,1].max(axis=1)))


class _Rows:
    """Traverse only candidate rows, retaining the exact source primitive context."""
    def __init__(self,array,projection,kind,original,ordinals):
        self.array=array;self.projection=projection;self.kind=kind;self.original=original;self.ordinals=ordinals
    def __len__(self):return len(self.array)
    def __iter__(self):
        for local,row in enumerate(self.array):
            self.projection.current=(self.kind,int(self.original[local]),int(self.ordinals[local]))
            yield row


class _Mesh:
    def __init__(self,original,world,triangles,lines,points,mapping,line_mapping):
        self.original=original;self.positions=world;self.displacements=None
        self.triangles=triangles;self.lines=lines;self.point_indices=points
        self.triangle_to_element=mapping;self.line_to_element=line_mapping
    @property
    def triangle_count(self):return len(self.triangles)
    def __getattr__(self,name):return getattr(self.original,name)


class _Handle:
    def __init__(self,original,mesh,chunks):self.original=original;self.mesh=mesh;self.chunks=chunks
    deformation_scale=0.
    transform=np.eye(4)
    def __getattr__(self,name):return getattr(self.original,name)


class _Projection(RetainedViewer):
    """Run the exact shared clipping/culling/binding service on compact candidates."""
    def __init__(self,owner):
        self.owner=owner;self.entries={};self._selection_index=None;self._selection_index_key=None
        self.current=None;self.ordinals=[]
    def __getattr__(self,name):return getattr(self.owner,name)
    def _selection_key(self,viewport):return None
    def _display_entries(self):return self.entries
    def _selection_binding(self,handle,entry,kind,primitive,*,chunk_id=None):
        return self.owner._selection_binding(handle.original,entry,kind,self.current[1],chunk_id=chunk_id)
    def _semantic_binding_visible(self,binding):
        accepted=self.owner._semantic_binding_visible(binding)
        if accepted:self.ordinals.append(self.current[2])
        return accepted


class PointQuery:
    def __init__(self,owner):
        self.owner=owner;self.world={};self.key=None;self.records=[]
        self.candidate_primitives=0;self.projected_primitives=0;self.camera_builds=0

    def close(self):
        self.world.clear();self.records.clear();self.key=None;self.owner=None

    def _world(self,handle,chunk_id,mesh):
        token=(id(handle),chunk_id)
        signature=(id(mesh.positions),id(mesh.displacements),id(mesh.triangles),id(mesh.lines),id(mesh.point_indices),
                   handle.transform.tobytes(),handle.deformation_scale)
        old=self.world.get(token)
        if old is not None and old['signature']==signature:return old
        points=mesh.positions
        if mesh.displacements is not None and handle.deformation_scale:
            points=points+handle.deformation_scale*mesh.displacements
        record=dict(signature=signature,positions=points@handle.transform[:3,:3].T+handle.transform[:3,3],mesh=mesh,visibility_key=None)
        self.world[token]=record
        return record

    def _visible(self,world_record,handle,entry,chunk_id,kind,count):
        owner=self.owner;state=owner._visibility_state
        if state.is_default:return np.ones(count,dtype=bool)
        key=(state,handle.generations.selection,handle.generations.topology,id(entry.get('owners')),id(entry.get('owner_resolver')),entry.get('appearance',{}).get('pickable',True))
        if world_record['visibility_key']!=key:
            world_record['visibility_key']=key;world_record['visibility']={}
        cache=world_record['visibility']
        if kind not in cache:
            cache[kind]=np.fromiter((owner._semantic_binding_visible(owner._selection_binding(handle,entry,kind,i,chunk_id=chunk_id) if entry['appearance'].get('pickable',True) else None) for i in range(count)),dtype=bool,count=count)
        return cache[kind]

    def _build(self):
        owner=self.owner;width,height=owner.framebuffer_size();camera=owner.camera
        key=(owner._selection_key((width,height)),camera.world_up.to_tuple())
        if key==self.key:return
        right,up,forward=camera.basis();basis=np.array([v.to_tuple() for v in (right,up,forward)])
        origin=np.array(camera.position.to_tuple());near,far=camera.near,camera.far;plane=owner._effective_section
        clipping=plane is not None and plane.enabled
        records=[];offset=0;live=set()
        for entry in owner._display_entries().values():
            handle=entry['handle']
            if handle.removed or not handle.visible:continue
            for chunk_id,mesh in [(None,handle.mesh),*handle.chunks]:
                live.add((id(handle),chunk_id));cached=self._world(handle,chunk_id,mesh);world=cached['positions']
                camera_points=(world-origin)@basis.T;screen=_screen(camera_points,width,height,camera.fov)
                distances=np.stack([world@np.array(p.normal.to_tuple())-p.offset for p in plane.planes],axis=-1) if clipping else None
                kinds=[]
                for kind,indices in (('triangle',mesh.triangles),('line',mesh.lines),('point',None if mesh.point_indices is None else mesh.point_indices[:,None])):
                    if indices is None or not len(indices):continue
                    if kind=='line' and not owner.show_mesh_lines and entry['appearance'].get('mesh_lines'):continue
                    count=len(indices);admitted=np.ones(count,dtype=bool)
                    depths=camera_points[indices,2]
                    if kind=='point':
                        admitted &= (depths[:,0]>=near)&(depths[:,0]<far)
                    else:
                        admitted &= (depths.max(axis=1)>=near)&(depths.min(axis=1)<=far)
                    if kind=='triangle':
                        mapping=np.arange(count,dtype=np.uint32) if mesh.triangle_to_element is None else mesh.triangle_to_element
                        if mesh.active_elements is not None:admitted &= mesh.active_elements[mapping]
                    elif kind=='line' and mesh.line_to_element is not None and mesh.active_elements is not None:
                        admitted &= mesh.active_elements[mesh.line_to_element]
                    point_distances=None if distances is None else distances[indices]
                    if clipping:
                        admitted &= (point_distances.max(axis=1)>=( -1e-9 if kind=='point' else 0.)).all(axis=-1)
                    boxes=_bbox(screen[indices])
                    inside=((depths>=near)&(depths<=far)).all(axis=1)
                    if clipping:inside &= (point_distances>=0.).all(axis=(1,2))
                    partial=np.flatnonzero(admitted & ~inside) if kind!='point' else np.empty(0,dtype=int)
                    if kind=='triangle' and entry['appearance'].get('cull_backface',True):
                        p=screen[indices]
                        area=(p[:,0,0]*p[:,1,1]-p[:,1,0]*p[:,0,1])+(p[:,1,0]*p[:,2,1]-p[:,2,0]*p[:,1,1])+(p[:,2,0]*p[:,0,1]-p[:,0,0]*p[:,2,1])
                        admitted[inside] &= area[inside]<-1e-9
                    for original in partial:
                        points=world[indices[original]]
                        if clipping:
                            result=plane.clip_polygon(points) if kind=='triangle' else plane.clip_segment(*points)
                            if not result:admitted[original]=False;continue
                            points=np.array([point.to_tuple() for point in result])
                        polygon=(points-origin)@basis.T
                        if kind=='triangle':
                            polygon=owner._far_clip(owner._near_clip(polygon,near),far)
                            if len(polygon)<3:admitted[original]=False;continue
                        else:
                            for boundary,is_near in ((near,True),(far,False)):
                                admitted_end=polygon[:,2]>=boundary if is_near else polygon[:,2]<=boundary
                                if not admitted_end.any():polygon=np.empty((0,3));break
                                if not admitted_end.all():
                                    amount=(boundary-polygon[0,2])/(polygon[1,2]-polygon[0,2])
                                    polygon[0 if not admitted_end[0] else 1]=polygon[0]+amount*(polygon[1]-polygon[0])
                            if not len(polygon):admitted[original]=False;continue
                        projected=_screen(polygon,width,height,camera.fov)
                        if kind=='triangle' and entry['appearance'].get('cull_backface',True):
                            area=sum(projected[i,0]*projected[(i+1)%len(projected),1]-projected[(i+1)%len(projected),0]*projected[i,1] for i in range(len(projected)))
                            if area>=-1e-9:admitted[original]=False;continue
                        boxes[original]=[projected[:,0].min(),projected[:,1].min(),projected[:,0].max(),projected[:,1].max()]
                    admitted &= self._visible(cached,handle,entry,chunk_id,kind,count)
                    ordinals=np.cumsum(admitted,dtype=np.int64)-1+offset
                    offset+=int(np.count_nonzero(admitted))
                    kinds.append(dict(kind=kind,indices=indices,boxes=boxes,admitted=admitted,ordinals=ordinals))
                records.append(dict(handle=handle,entry=entry,chunk_id=chunk_id,mesh=mesh,world=world,kinds=kinds))
        for token in set(self.world)-live:self.world.pop(token)
        self.key=key;self.records=records;self.camera_builds+=1

    def query(self,x,y,selection_filter,radius):
        self._build();owner=self.owner;projection=_Projection(owner);grouped={};total=0
        ratio=owner.pixel_ratio
        for record in self.records:
            style=record['entry']['appearance']
            if style.get('opacity',1)<=0:continue
            selected={}
            for group in record['kinds']:
                kind=group['kind'];pad=radius
                if kind=='line':pad+=max(2.5,float(style.get('line_width',1))*ratio*.5)
                elif kind=='point':pad+=max(3.5,float(style.get('point_size',6))*ratio*.5)
                b=group['boxes'];keep=group['admitted']&(b[:,0]<=x+pad)&(b[:,2]>=x-pad)&(b[:,1]<=y+pad)&(b[:,3]>=y-pad)
                ids=np.flatnonzero(keep);selected[kind]=(group,ids);total+=len(ids)
            if not any(len(ids) for group,ids in selected.values()):continue
            nodes=[group['indices'][ids].ravel() for group,ids in selected.values() if len(ids)]
            unique=np.unique(np.concatenate(nodes))
            rows={};mapping=None;line_mapping=None
            for kind in ('triangle','line','point'):
                if kind not in selected:continue
                group,ids=selected[kind]
                array=np.searchsorted(unique,group['indices'][ids])
                if kind=='point':array=array[:,0]
                rows[kind]=_Rows(array,projection,kind,ids,group['ordinals'][ids])
                mesh=record['mesh']
                if kind=='triangle':mapping=ids.astype(np.uint32) if mesh.triangle_to_element is None else mesh.triangle_to_element[ids]
                elif kind=='line' and mesh.line_to_element is not None:line_mapping=mesh.line_to_element[ids]
            triangles=rows.get('triangle',_Rows(np.empty((0,3),dtype=np.uint32),projection,'triangle',[],[]))
            candidate_mesh=_Mesh(record['mesh'],record['world'][unique],triangles,rows.get('line'),rows.get('point'),mapping,line_mapping)
            token=id(record['handle'])
            group=grouped.setdefault(token,dict(handle=record['handle'],entry=record['entry'],main=None,chunks=[]))
            if record['chunk_id'] is None:group['main']=candidate_mesh
            else:group['chunks'].append((record['chunk_id'],candidate_mesh))
        self.candidate_primitives=total
        if not grouped:self.projected_primitives=0;return ()
        for token,group in grouped.items():
            main=group['main']
            if main is None:
                original=group['handle'].mesh
                main=_Mesh(original,np.empty((0,3)),_Rows(np.empty((0,3),dtype=np.uint32),projection,'triangle',[],[]),None,None,np.empty(0,dtype=np.uint32),None)
            proxy=_Handle(group['handle'],main,group['chunks'])
            projection.entries[token]=dict(group['entry'],handle=proxy)
        index=projection._projected_selection_index()
        if len(index.primitives)!=len(projection.ordinals):
            raise RuntimeError('canonical point projection traversal mismatch')
        styles={entry['item']:entry['appearance'] for entry in owner._display_entries().values()}
        primitives=[]
        for primitive,ordinal in zip(index.primitives,projection.ordinals):
            style=styles[primitive.item]
            pad=max(primitive.radius,float(style.get('line_width',1))*ratio*.5) if primitive.shape=='segment' else primitive.radius
            primitives.append(replace(primitive,index=ordinal,radius=pad))
        self.projected_primitives=len(primitives)
        return ProjectedSelectionIndex(primitives,index.width,index.height).point_hits(x,y,selection_filter,radius=radius)
