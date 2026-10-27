"""Viewer-owned section-box handles; every drag changes visual clipping only."""
import math
import numpy as np
from .core import as_point

class SectionBoxGizmo:
    def __init__(self,viewer): self.viewer=viewer;self.drag=None;self.handles=[]
    def draw(self,hud):
        v=self.viewer;box=v.section_box;self.handles=[]
        if box is None or not box.enabled:return
        c=np.array(box.center.to_tuple());e=np.array(box.half_extents.to_tuple())
        axes=np.array([box.planes[2*i].normal.to_tuple() for i in range(3)])
        ratio=v.pixel_ratio
        def project(p):
            q=v.project_point(as_point(p))
            return None if q is None else (q[0]/ratio,q[1]/ratio)
        corners=[c+np.array(signs)*e@axes for signs in __import__('itertools').product((-1,1),repeat=3)]
        for i,a in enumerate(corners):
            for j,b in enumerate(corners[:i]):
                if abs(np.linalg.norm((a-b)@axes.T/(2*e))-1.)<1e-7:
                    pa,pb=project(a),project(b)
                    if pa and pb:hud.line(pa,pb,'#20a6c7',width=1.5)
        for i,axis in enumerate(axes):
            for sign in (-1,1):
                q=project(c+axis*e[i]*sign)
                if q:
                    self.handles.append((q,'face',i,sign));hud.circle(q,5,'#007e9c',width=2)
        center=project(c)
        if center:
            self.handles.append((center,'translate',0,1))
            hud.circle(center,7,'#007e9c',width=2)
            # Distinct screen-space controls stay available even when a local
            # axis points directly at the camera. Drag around box center;
            # each control composes about the named local (not camera) axis.
            for axis,(offset,color,label) in enumerate((((44,0),'#c83737','RX'),((0,-44),'#16833b','RY'),((-34,-34),'#285bd4','RZ'))):
                point=(center[0]+offset[0],center[1]+offset[1])
                self.handles.append((point,'rotate',axis,1))
                hud.circle(point,8,color,width=2)
                hud.text((point[0]+10,point[1]-7),label,color)
    def press(self,event):
        ratio=self.viewer.pixel_ratio;p=np.array([event.x/ratio,event.y/ratio])
        candidates=[h for h in self.handles if np.linalg.norm(p-np.array(h[0]))<=10]
        if not candidates:return False
        handle=min(candidates,key=lambda h:np.linalg.norm(p-np.array(h[0])))
        self.drag=(handle,p,self.viewer.section_box)
        return True
    def motion(self,event):
        if self.drag is None:return False
        v=self.viewer;(handle,start,box)=self.drag
        p=np.array([event.x,event.y])/v.pixel_ratio;delta=p-start
        right,up,forward=v.camera.basis()
        distance=max((box.center-v.camera.position).length(),v.camera.near)
        unit=2*distance*math.tan(v.camera.fov/2)/(v.framebuffer_size()[1]/v.pixel_ratio)
        world=np.array(right.to_tuple())*delta[0]*unit-np.array(up.to_tuple())*delta[1]*unit
        center=np.array(box.center.to_tuple());extents=np.array(box.half_extents.to_tuple());orientation=box.orientation
        if handle[1]=='translate':center+=world
        elif handle[1]=='face':
            i,sign=handle[2:];axis=np.array(box.planes[i*2].normal.to_tuple())
            amount=float(world@axis)*sign
            new_extent=max(extents[i]+amount/2,1e-8)
            center+=axis*sign*(new_extent-extents[i]);extents[i]=new_extent
        else:
            projected=v.project_point(box.center)
            pivot=np.array(projected[:2])/v.pixel_ratio
            first=start-pivot;current=p-pivot
            angle=math.atan2(first[0]*current[1]-first[1]*current[0],float(first@current))
            axis=np.eye(3)[handle[2]]
            # q_start * q_local: selected axis follows the box's original
            # orientation, preserving arbitrary normalized xyzw rotations.
            a=np.array(orientation);b=np.array([*(axis*math.sin(angle/2)),math.cos(angle/2)])
            orientation=(*tuple(a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3])),a[3]*b[3]-a[:3]@b[:3])
        v.set_section_box(center,extents,orientation)
        return True
    def release(self):
        active=self.drag is not None;self.drag=None;return active
    def cancel(self):
        if self.drag is None:return
        box=self.drag[2];self.drag=None
        self.viewer.set_section_box(box.center,box.half_extents,box.orientation,box.enabled)
