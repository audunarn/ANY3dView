"""Renderer adapters for existing retained chunk identities; no copied geometry."""
class ChunkView:
    def __init__(self,parent,mesh):self.parent=parent;self.mesh=mesh
    def __getattr__(self,name):return getattr(self.parent,name)
    @property
    def chunks(self):return ()

def expand_chunks(renderer,entries,masks,chunk_masks):
    expanded={};expanded_masks=dict(masks);live=set()
    for key,entry in entries.items():
        handle=entry['handle'];expanded[key]=entry
        for chunk_id,mesh in handle.chunks:
            token=(id(handle),chunk_id);live.add(token)
            proxy=renderer.chunk_views.get(token)
            if proxy is None:
                proxy=ChunkView(handle,mesh);renderer.chunk_views[token]=proxy
            proxy.mesh=mesh
            appearance=dict(entry['appearance'],face_colors=None)
            expanded[id(proxy)]=dict(entry,handle=proxy,appearance=appearance)
            expanded_masks[id(proxy)]=chunk_masks.get(token,{})
    for token in set(renderer.chunk_views)-live:renderer.chunk_views.pop(token)
    return expanded,expanded_masks


def stipple_window(value,phase=0):
    """Same renderer-neutral Bayer window used by the existing OpenGL path."""
    import re
    specification=str(value or '')
    if not specification:return 0,64,0
    count={'gray12':8,'gray25':16,'gray50':32,'gray75':48}.get(specification.casefold())
    start=0;rotation=(int(phase)%4)*16
    generated=re.search(r'anytk3d_(\d{2})_(\d{2})_(\d{2})',specification.casefold())
    if generated is not None:start,count,rotation=(int(value) for value in generated.groups())
    elif count is None:count=32
    start=max(0,min(63,int(start)));count=max(1,min(64-start,int(count)))
    return start,count,int(rotation)&63

BAYER=(0,32,8,40,2,34,10,42,48,16,56,24,50,18,58,26,
       12,44,4,36,14,46,6,38,60,28,52,20,62,30,54,22,
       3,35,11,43,1,33,9,41,51,19,59,27,49,17,57,25,
       15,47,7,39,13,45,5,37,63,31,55,23,61,29,53,21)

def stipple_visible(value,phase,x,y,height):
    start,count,rotation=stipple_window(value,phase)
    threshold=(BAYER[((int(height)-1-int(y))&7)*8+(int(x)&7)]-rotation)&63
    return start<=threshold<start+count
