"""Opt-in Qt WebGPU host. Presentation is explicit GPU RGBA readback to QWidget."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget, QVBoxLayout
from .qt import QtViewer, _InputSurface, _SoftwareResources, _PainterHud
from .retained import MeshHandle
from .capabilities import ViewerCapabilities
from .errors import GPUUnavailableError
from .webgpu import WebGPURenderer


def validate_capture_admission(renderer,width,height,*,shadows=False):
    """Refuse unsupported or runaway temporary export targets before allocation."""
    limits=renderer.device.limits
    # wgpu-native 0.32 exposes hyphenated keys; accept underscore mappings
    # as well for compatible device adapters and explicit test doubles.
    limit=limits.get('max_texture_dimension_2d',limits.get('max-texture-dimension-2d'))
    if limit is None or int(limit)<1:
        raise ValueError('Device texture dimension limit is unavailable; reduce resolution or select a supported renderer')
    if max(width,height)>int(limit):
        raise ValueError(f'Capture exceeds device 2D texture limit {limit}; reduce resolution or anti-aliasing')
    samples=int(renderer.samples)
    bytes_per_pixel=24+4*samples+(24*samples if samples>1 else 0)
    if renderer.ao_quality!='Off' or shadows:bytes_per_pixel+=32
    required=int(width)*int(height)*bytes_per_pixel
    if required>2*1024**3:
        raise ValueError(f'Capture attachments need {required/1024**2:.0f} MiB, exceeding the 2 GiB temporary budget; reduce resolution or anti-aliasing')
    return required


def _unsupported(*args, **kwargs):
    raise NotImplementedError("experimental WebGPU supports opaque indexed MeshArrays only; this feature is unsupported")


class _WebGPUHandle(MeshHandle):
    __slots__ = ()

    def _validate_chunk(self,mesh):
        from .arrays import MeshArrays
        from .webgpu.renderer import validate_gpu_mesh,scalar_payload
        self._check()
        if not isinstance(mesh,MeshArrays):raise TypeError("chunk must be MeshArrays")
        validate_gpu_mesh(mesh,self.deformation_scale);scalar_payload(mesh,self.appearance)

    def add_chunk(self,chunk_id,chunk,**ownership):
        self._validate_chunk(chunk)
        return super().add_chunk(chunk_id,chunk,**ownership)

    def replace_chunk(self,chunk_id,replacement,**ownership):
        self._validate_chunk(replacement)
        return super().replace_chunk(chunk_id,replacement,**ownership)

    def _replace(self, field, value, generation):
        from dataclasses import replace
        from .webgpu.renderer import scalar_payload
        self._check()
        candidate=replace(self.mesh, **{field:value})
        from .webgpu.renderer import validate_gpu_mesh
        validate_gpu_mesh(candidate,self.deformation_scale)
        scalar_payload(candidate,self.appearance)
        return super()._replace(field,value,generation)

    def update_scalar_fields(self, *, node_scalars=None, element_scalars=None, before_publish=None):
        from .webgpu.renderer import scalar_payload
        def reserve(candidate):
            scalar_payload(candidate,self.appearance)
            if before_publish is not None: before_publish(candidate)
        return super().update_scalar_fields(node_scalars=node_scalars,
            element_scalars=element_scalars,before_publish=reserve)

    def update_appearance(self, **changes):
        from .appearance import validated_appearance
        from .webgpu.renderer import scalar_payload
        self._check()
        candidate = validated_appearance(changes.get("scalar_range",self.appearance.scalar_range),
            changes.get("colormap",self.appearance.colormap))
        scalar_payload(self.mesh,candidate)
        for _,mesh in self.chunks:scalar_payload(mesh,candidate)
        return super().update_appearance(**changes)

    def set_deformation_scale(self, scale):
        import numpy as np
        self._check()
        if abs(float(scale)) > np.finfo(np.float32).max:
            raise ValueError("WebGPU deformation scale exceeds float32")
        from .webgpu.renderer import validate_gpu_mesh
        validate_gpu_mesh(self.mesh,scale)
        for _,mesh in self.chunks:validate_gpu_mesh(mesh,scale)
        return super().set_deformation_scale(scale)

    def _check(self):
        super()._check()
        owner = getattr(self._callback, "__self__", None)
        if owner is not None and owner._closed:
            raise RuntimeError("retained mesh belongs to a closed WebGPU viewer")

    def remove(self):
        if self.removed:
            return
        self._check()
        return super().remove()

    def set_transform(self, transform):
        self._check()
        import numpy as np
        matrix = np.asarray(transform)
        if matrix.shape == (4, 4) and not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise NotImplementedError("WebGPU supports affine transforms only")
        return super().set_transform(transform)


class _Metadata(_SoftwareResources):
    """Shared CPU selection masks, explicitly not a renderer."""
    def add_mesh(self, handle, **appearance):
        handle.__class__ = _WebGPUHandle
        super().add_mesh(handle, **appearance)


class _BitmapSurface(_InputSurface, QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, event):
        owner = self.owner
        if owner._closed:
            return
        try:
            pixels = owner._render_pixels()
            height, width, _ = pixels.shape
            image = QImage(pixels.data, width, height, width * 4, QImage.Format_RGBA8888)
            painter = QPainter(self)
            painter.drawImage(self.rect(), image)
            painter.scale(1/owner.pixel_ratio,1/owner.pixel_ratio)
            owner._hud.painter=painter
            owner._render_hud(owner.framebuffer_size())
            owner._hud.painter=None
            painter.end()
        except Exception as error:
            owner._last_render_error = f"{type(error).__name__}: {error}"
            owner.render_error.emit(owner._last_render_error)


class QtWebGPUViewer(QtViewer):
    """Experimental opaque array viewer. Does not imply full QtViewer parity.

    MeshHandle and retained scene identities, CPU semantic picking, camera/input
    and scheduler are shared with the other hosts. Display rasterization is GPU;
    RGBA pixels are synchronously read back for portable QWidget presentation.
    """
    def __init__(self, parent=None, **options):
        unknown = set(options) - {"width", "height", "bg", "interactive_fps", "shading", "interaction_profile"}
        if unknown:
            raise TypeError(f"unsupported WebGPU options: {sorted(unknown)}")
        from .core import parse_color
        if parse_color(options.get("bg", "white")) is None:
            raise ValueError("unsupported background color")
        QWidget.__init__(self, parent)
        from threading import get_ident
        self._owner_thread = get_ident()
        self.initialize_state(**dict({"shading":False}, **options))
        self._backend = "webgpu"
        from .display import Contours
        self._contours=Contours()
        self._selection_outline=(True,False)
        self._shadows=("Off",.5)
        self._cut_caps=(False,True)
        self._cap_provider=None
        self._cap_handles=[]
        self._cap_key=None
        self._present_method="bitmap"
        self._present_error=None
        from .section_gizmo import SectionBoxGizmo
        self._section_gizmo=SectionBoxGizmo(self)
        self._capture_size=None
        self._host = self
        self._initializing = True
        self._timers = set()
        self._hover_timer = self._pending_hover = self._update_notifier = None
        self._software_prep = {}
        self._software_revision = 0
        self._last_render_error = None
        self._renderer = _Metadata()
        self._hud = _PainterHud()
        self._show_axis_indicator = self.show_axis_ruler = False
        self.show_mesh_lines = True
        self._occlude_lines = True
        try:
            self._webgpu = WebGPURenderer()
        except Exception as error:
            self._update_scheduler.close()
            self.deleteLater()
            raise GPUUnavailableError(str(error), diagnostics=(type(error).__name__, str(error))) from error
        try:
            self.surface = self.canvas = _BitmapSurface(self)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.surface)
            self.resize(self.width, self.height)
            self._start_update_pump()
            self._initializing = False
        except BaseException:
            self.destroy()
            raise

    @property
    def capabilities(self):
        return ViewerCapabilities(gpu=True, dynamic_arrays=True, camera_controls=True,
            image_capture=True, semantic_selection=True, semantic_visibility=True,
            through_selection=True, hover_selection=True, region_selection=True,
            lasso_selection=True, line_occlusion=True, smooth_normals=True, face_color_update=True,
            clipping_planes=True, transparency=True, stippled_transparency=True,
            legacy_primitives=True,text_hud=True,legends=True,animation=True,incremental_chunks=True,
            viewer_commands=True,command_history=True,work_plane_projection=True,node_scalar_field=True,
            element_scalar_field=True, shader_deformation=True, active_element_mask=True)

    @property
    def renderer_diagnostics(self):
        renderer = self._webgpu
        return {"draw_calls": renderer.draw_calls, "frame_count": renderer.frame_count,
                "geometry_uploads": renderer.geometry_uploads,
                "retained_meshes": len(self._entries), "hud_uploads": self._hud.uploads,
                "index_uploads": renderer.index_uploads, "field_uploads": renderer.field_uploads,
                "palette_updates": renderer.palette_updates,
                "samples":renderer.samples,"background_gradient":renderer.gradient,
                "feature_edges":renderer.feature_edges,"ambient_occlusion":renderer.ao_quality,
                "effective_ambient_occlusion":renderer.ao_effective,"ao_passes":renderer.ao.passes,
                "shadows":self._shadows[0],"shadow_method":"screen-space-directional",
                "contours":self._contours.mode,"selection_outline":self._selection_outline,
                "present_method":self._present_method,"present_error":self._present_error}

    def diagnostic_snapshot(self):
        renderer = self._webgpu
        return {"schema": "any3dview.qt.webgpu", "version": 1,
                "active_backend": "webgpu", "closed": self._closed,
                "device": dict(renderer.info), "present_method": self._present_method,
                "present_error": self._present_error,
                "native_promotion": False,
                "native_presented_frames":getattr(self.surface,"presented_frames",0),
                "native_surface_format":getattr(self.surface,"format",None),
                "integer_picking": False, "selection_method": "shared-cpu-semantic",
                "frame_count": renderer.frame_count, "draw_calls": renderer.draw_calls,
                "geometry_uploads": renderer.geometry_uploads,
                "index_uploads": renderer.index_uploads, "field_uploads": renderer.field_uploads,
                "palette_updates": renderer.palette_updates,
                "last_render_error": self._last_render_error}

    def _render_pixels(self, *, readback=True):
        self.make_current()
        self._refresh_caps()
        return self._webgpu.render(self._display_entries(), self._renderer.masks,
            self.camera, self.framebuffer_size(), self.bg, self._highlight_fill,
            light=self._light,shading=self._shading_enabled,section=self._effective_section,
            pixel_ratio=self.pixel_ratio,occlude_lines=self._occlude_lines,chunk_masks=self._renderer.chunk_masks,mesh_lines=self.show_mesh_lines,
            interactive=bool(self._drag or self._wheel_finish_after_id is not None),
            contours=self._contours,selection_outline=self._selection_outline,
            shadows=self._shadows,readback=readback)

    def set_present_method(self, method='bitmap'):
        self.make_current()
        if method not in ('bitmap','native','auto'): raise ValueError('present method must be bitmap, native or auto')
        effective='bitmap' if method=='auto' else method
        if effective==self._present_method: return
        if effective=='native':
            from .webgpu.presentation import create_native_surface
            try: candidate=create_native_surface(self)
            except Exception as error:
                self._present_error=f'{type(error).__name__}: {error}'
                raise GPUUnavailableError('native presentation unavailable: '+self._present_error) from error
        else: candidate=_BitmapSurface(self)
        previous=self.surface
        self.layout().replaceWidget(previous,candidate)
        self.surface=self.canvas=candidate
        self._present_method=effective
        previous.hide()
        if hasattr(previous,'release'):previous.release()
        previous.deleteLater();candidate.show();self.redraw()

    def set_contours(self, mode='Continuous', levels=(), zero_contour=False, threshold=None):
        self.make_current()
        from .display import Contours,contour_uniforms
        if threshold is not None:
            if not isinstance(threshold,dict) or set(threshold)!={'direction','value'}:
                raise ValueError('threshold requires direction and value')
            threshold=(threshold['direction'],threshold['value'])
        candidate=Contours(mode,tuple(levels),bool(zero_contour),threshold)
        for entry in self._display_entries().values():
            from .webgpu.renderer import scalar_payload
            contour_uniforms(candidate,scalar_payload(entry['handle'].mesh,entry['handle'].appearance)[1])
        self._contours=candidate
        self.redraw()

    def set_selection_outline(self, enabled=True, through_geometry=False):
        self.make_current()
        self._selection_outline=(bool(enabled),bool(through_geometry))
        self.redraw()

    def set_shadows(self, quality='Off', strength=.5):
        import math
        self.make_current()
        strength=float(strength)
        if quality not in ('Off','Low','High') or not math.isfinite(strength) or not 0<=strength<=1:
            raise ValueError('shadows require Off/Low/High and strength between zero and one')
        self._shadows=(quality,strength)
        self.redraw()

    def set_cut_caps(self, enabled=True, hatch=True, provider=None):
        self.make_current()
        candidate=provider if provider is not None else self._cap_provider
        if enabled and not callable(candidate):
            raise NotImplementedError('cut caps require an owner-qualified closed-volume cap provider; open or unqualified display surfaces are unavailable')
        previous=self._cut_caps,self._cap_provider,self._cap_key
        self._cut_caps=(bool(enabled),bool(hatch));self._cap_provider=candidate;self._cap_key=None
        try:self._refresh_caps()
        except BaseException:
            self._cut_caps,self._cap_provider,self._cap_key=previous
            raise
        self.redraw()

    def clear_cut_cap_provider(self):
        """Remove derived caps and release their application-owned provider."""
        self.make_current()
        self.set_cut_caps(False,hatch=self._cut_caps[1])
        self._cap_provider=None
        self._cap_key=None

    def _refresh_caps(self):
        from .clipping import CutCap
        from .arrays import MeshArrays
        section=self._effective_section
        cap_ids={id(h) for h in self._cap_handles}
        source_handles=[e['handle'] for e in self._entries.values() if id(e['handle']) not in cap_ids]
        key=(self._cut_caps,id(self._cap_provider),None if section is None else section.key,
             tuple((id(h),id(h.mesh),h.visible,h.removed,h.deformation_scale,h.transform.tobytes()) for h in source_handles))
        if key==self._cap_key:return
        payload=tuple(self._cap_provider(self,section.planes)) if self._cut_caps[0] and section is not None else ()
        payload=tuple(cap for cap in payload if not isinstance(cap,CutCap) or not getattr(cap.source_handle,'removed',False))
        for cap in payload:
            if not isinstance(cap,CutCap):raise TypeError('cap provider must return CutCap objects')
            if not any(cap.source_handle is h and not h.removed for h in source_handles):raise ValueError('cap source must be a live source handle in this viewer')
            mesh=cap.mesh
            if not isinstance(mesh,MeshArrays) or mesh.triangle_count==0 or mesh.displacements is not None or mesh.node_scalars is not None or mesh.element_scalars is not None:
                raise ValueError('caps require neutral triangle MeshArrays without scientific fields or displacements')
            if cap.owners is None:raise ValueError('cap source semantic owners are required')
            self._validate_mesh_support(mesh)
        payload=tuple(cap for cap in payload if cap.source_handle.visible)
        candidates=[]
        try:
            for cap in payload:
                h=self.add_mesh_arrays(cap.mesh,color='#e3ddd1',line_color='#6b6252',cull_backface=False,lit=False,owners=cap.owners)
                candidates.append(h)
                self._entries[id(h)]['appearance']['_cut_cap_hatch']=self._cut_caps[1]
        except BaseException:
            for h in candidates:h.remove()
            raise
        for h in self._cap_handles:h.remove()
        self._cap_handles=candidates;self._cap_key=key


    def framebuffer_size(self):
        return getattr(self,"_capture_size",None) or super().framebuffer_size()

    def capture_image(self, width=None, height=None):
        import math
        current=super().framebuffer_size()
        for value in (width,height):
            if value is not None and (isinstance(value,bool) or int(value)!=value or not 1<=int(value)<=16384):
                raise ValueError('capture dimensions must be integers between 1 and 16384')
        if width is None and height is None:
            validate_capture_admission(self._webgpu,*current,shadows=self._shadows[0]!='Off')
            return self._capture_image_current()
        if width is None:width=max(1,round(height*current[0]/current[1]))
        if height is None:height=max(1,round(width*current[1]/current[0]))
        if max(width,height)>16384:raise ValueError('capture dimensions exceed 16384; reduce resolution')
        validate_capture_admission(self._webgpu,width,height,shadows=self._shadows[0]!='Off')
        previous=self._capture_size
        renderer=self._webgpu
        names=('size','target','depth','opaque','accumulation','reveal','multisampled')
        saved={name:getattr(renderer,name,None) for name in names}
        renderer.size=None;renderer.multisampled={}
        for name in names[1:-1]:setattr(renderer,name,None)
        try:
            self._capture_size=(int(width),int(height))
            return self._capture_image_current()
        finally:
            self._capture_size=previous
            for texture in renderer.multisampled.values():texture.destroy()
            for name in names[1:-1]:
                texture=getattr(renderer,name,None)
                if texture is not None:texture.destroy()
            for name,value in saved.items():setattr(renderer,name,value)

    def _capture_image_current(self):
        from PIL import Image
        import numpy as np
        pixels=self._render_pixels()
        height,width,_=pixels.shape
        image=QImage(pixels.data,width,height,width*4,QImage.Format_RGBA8888).copy()
        painter=QPainter(image)
        try:
            self._hud.painter=painter
            self._render_hud((width,height))
        finally:
            self._hud.painter=None
            painter.end()
        return Image.fromarray(np.asarray(image.constBits()).reshape(height,width,4).copy(),"RGBA")

    def make_current(self):
        from threading import get_ident
        if get_ident() != self._owner_thread:
            raise RuntimeError("WebGPU viewer updates must run on the owning thread")
        if self._closed:
            raise RuntimeError("Cannot update a closed WebGPU viewer")

    def _validate_mesh_support(self, mesh, **kwargs):
        from .webgpu.renderer import validate_gpu_mesh
        validate_gpu_mesh(mesh)

    def add_mesh_arrays(self, mesh, *, lit=False, **options):
        return super().add_mesh_arrays(mesh, lit=lit, **options)

    def _add_mesh_arrays(self, mesh, **options):
        # Refuse before shared registration or mutation. Explicit unsupported
        # appearance must never be accepted and silently omitted.
        self.make_current()
        self._validate_mesh_support(mesh)
        from .core import parse_color
        for key in ("color", "line_color", "invalid_color", "back_color"):
            if options.get(key) and parse_color(options[key]) is None:
                raise ValueError(f"invalid {key}")
        from .appearance import validated_appearance
        from .webgpu.renderer import scalar_payload
        scalar_payload(mesh,validated_appearance(options.get("scalar_range"),options.get("colormap")))
        if options.get("face_colors") is not None:
            from .face_colors import normalize_face_colors
            options["face_colors"],_ = normalize_face_colors(options["face_colors"])
        accepted={"color","outline","width","layer","cull_backface","opacity","stipple","back_color","tags","lit","two_sided_shell","face_colors","scalar_range","colormap","invalid_color","owners","owner_resolver","line_color","line_width","point_color","point_outline","point_size","draw_overlay","pickable","depth_only","stipple_phase","_point_overlay","_mesh_outline","_outline_groups","_outline_vertex_ids"}
        unknown=set(options)-accepted
        if unknown:raise TypeError(f"unknown WebGPU appearance options: {sorted(unknown)}")
        import numpy as np
        for name in ("width","line_width","point_size"):
            if options.get(name) is not None and (not np.isfinite(float(options[name])) or float(options[name])<=0):
                raise ValueError(f"{name} must be positive and finite")
        opacity=float(options.get("opacity",1))
        if not np.isfinite(opacity) or not 0<=opacity<=1:raise ValueError("opacity must be within0..1")
        return super()._add_mesh_arrays(mesh, **options)

    def _scene_bounds(self):
        import numpy as np
        bounds=[]
        plane=self._section_plane
        entries=self._display_entries()
        if any(entry["handle"].chunks for entry in entries.values()):
            from .webgpu.scene import expand_chunks
            entries,_=expand_chunks(self._webgpu,entries,self._renderer.masks,self._renderer.chunk_masks)
        for entry in entries.values():
            handle=entry["handle"]
            if handle.removed or not handle.visible: continue
            mesh=handle.mesh
            points=np.asarray(mesh.positions,dtype=np.float64)
            if mesh.displacements is not None:
                points=points+handle.deformation_scale*mesh.displacements
            world=points@handle.transform[:3,:3].T+handle.transform[:3,3]
            groups=[]
            if len(mesh.triangles):
                mapping=np.arange(len(mesh.triangles)) if mesh.triangle_to_element is None else mesh.triangle_to_element
                admit=np.ones(len(mapping),dtype=bool) if mesh.active_elements is None else mesh.active_elements[mapping]
                triangles=mesh.triangles[admit]
                if plane is not None and plane.enabled:
                    for triangle in triangles:
                        clipped=plane.clip_polygon(world[triangle])
                        if clipped: bounds.append(np.array([point.to_tuple() for point in clipped]))
                else: groups.append(triangles.ravel())
            if mesh.lines is not None:
                admit=np.ones(len(mesh.lines),dtype=bool) if mesh.active_elements is None or mesh.line_to_element is None else mesh.active_elements[mesh.line_to_element]
                lines=mesh.lines[admit]
                if plane is not None and plane.enabled:
                    for line in lines:
                        clipped=plane.clip_segment(*world[line])
                        if clipped: bounds.append(np.array([point.to_tuple() for point in clipped]))
                else: groups.append(lines.ravel())
            if mesh.point_indices is not None:
                ids=mesh.point_indices
                if plane is not None and plane.enabled:
                    ids=np.asarray([node for node in ids if plane.contains(world[node])],dtype=np.uint32)
                if len(ids):bounds.append(world[ids])
            if groups:
                ids=np.unique(np.concatenate(groups))
                if len(ids): bounds.append(world[ids])
        if not bounds:return None
        points=np.concatenate(bounds)
        return points.min(axis=0),points.max(axis=0)

    def set_mesh_opacity(self,handle,opacity):
        self.make_current()
        result=super().set_mesh_opacity(handle,opacity)
        self._selection_index=None
        return result

    def set_mesh_style(self,handle,**values):
        self.make_current()
        result=super().set_mesh_style(handle,**values)
        if "width" in values or "opacity" in values:self._selection_index=None
        return result

    def set_background(self, color):
        from .core import parse_color
        if parse_color(color) is None:
            raise ValueError("unsupported background color")
        return super().set_background(color)

    def set_shading(self, enabled=True):
        self.make_current()
        return super().set_shading(enabled)

    def _release(self,event):
        refine=bool(self._drag and self._webgpu.ao_quality=='High')
        super()._release(event)
        if refine:self.redraw()

    def set_render_quality(self, *, samples=1, gradient=False, feature_edges=False,
                           ambient_occlusion='Off',ao_strength=.8,ao_radius=24.):
        """Configure display quality without replacing geometry or ownership."""
        self.make_current()
        if isinstance(samples,bool) or samples not in (1,4):raise ValueError('samples must be 1 or 4')
        from .webgpu.ssao import validated_settings
        ambient_occlusion,ao_strength,ao_radius=validated_settings(ambient_occlusion,ao_strength,ao_radius)
        renderer=self._webgpu
        renderer.samples=int(samples);renderer.gradient=bool(gradient);renderer.feature_edges=bool(feature_edges)
        renderer.ao_quality=ambient_occlusion;renderer.ao_strength=ao_strength;renderer.ao_radius=ao_radius
        self.redraw()

    def set_light(self, direction=None, ambient=None, diffuse=None, specular=None,
                  shininess=None, follow_camera=None, enabled=None):
        values=dict(direction=direction,ambient=ambient,diffuse=diffuse,specular=specular,
            shininess=shininess,follow_camera=follow_camera,enabled=enabled)
        import numpy as np
        from .core import as_point
        self.make_current()
        for key,value in values.items():
            if value is None: continue
            if key == "direction":
                if not np.isfinite(as_point(value).to_tuple()).all() or np.max(np.abs(as_point(value).to_tuple()))>np.finfo(np.float32).max:
                    raise ValueError("light direction must be finite")
            elif key in {"ambient","diffuse","specular","shininess"}:
                if not np.isfinite(float(value)) or abs(float(value))>np.finfo(np.float32).max:
                    raise ValueError("light coefficient must be finite and representable")
        return super().set_light(**values)

    def set_section_box(self, center, half_extents, orientation=(0.,0.,0.,1.), enabled=True):
        self.make_current()
        return super().set_section_box(center,half_extents,orientation,enabled)

    def clear_section_box(self):
        self.make_current()
        return super().clear_section_box()

    def set_section_plane(self, normal=(1.,0.,0.), offset=0., *, enabled=True):
        self.make_current()
        return super().set_section_plane(normal,offset,enabled=enabled)

    def clear_section_plane(self):
        self.make_current()
        return super().clear_section_plane()

    def set_axis_indicator(self,visible=True):
        self.make_current();return super().set_axis_indicator(visible)

    def set_axis_ruler(self,visible=True):
        self.make_current();return super().set_axis_ruler(visible)

    def set_mesh_lines(self,visible=True):
        self.make_current();return super().set_mesh_lines(visible)

    def set_occlude_lines(self,enabled=True):
        self.make_current();return super().set_occlude_lines(enabled)

    def apply_view_state(self, state, **kwargs):
        from .contracts import ViewerState
        from .core import parse_color
        import numpy as np
        self.make_current()
        if not isinstance(state, ViewerState): raise TypeError("state must be ViewerState")
        if parse_color(state.background) is None: raise ValueError("unsupported background color")
        from .clipping import SectionPlane
        if state.section_plane is not None and not isinstance(state.section_plane,SectionPlane):
            raise TypeError("section_plane must be SectionPlane or None")
        if state.light is not None and (not np.isfinite(state.light.key()).all() or np.max(np.abs(state.light.key()))>np.finfo(np.float32).max):
            raise ValueError("light must be finite")
        from .display import Contours
        if state.contours is not None and not isinstance(state.contours,Contours): raise TypeError('contours must be Contours')
        if state.cut_caps[0]: raise NotImplementedError('owner-qualified closed-volume cap provider required')
        import math
        if len(state.shadows)!=2 or state.shadows[0] not in ('Off','Low','High') or not math.isfinite(float(state.shadows[1])) or not 0<=state.shadows[1]<=1:
            raise ValueError('invalid portable shadow policy')
        if len(state.selection_outline)!=2:raise ValueError('invalid portable selection outline policy')
        result=super().apply_view_state(state, **kwargs)
        self._contours=state.contours or Contours()
        self._selection_outline=state.selection_outline
        self.set_cut_caps(False,hatch=state.cut_caps[1])
        self.set_shadows(*state.shadows)
        # Native is explicitly requested separately, after candidate validation.
        return result

    def _projected_selection_index(self):
        from dataclasses import replace
        from .selection import ProjectedSelectionIndex
        index=super()._projected_selection_index()
        if index is getattr(self,"_webgpu_pick_index",None):return index
        styles={entry["item"]:entry["appearance"] for entry in self._display_entries().values()}
        primitives=[]
        for primitive in index.primitives:
            style=styles.get(primitive.item,{})
            if style.get("opacity",1)<=0:continue
            if primitive.shape=="segment":
                primitive=replace(primitive,radius=max(primitive.radius,float(style.get("line_width",1))*self.pixel_ratio*.5))
            primitives.append(primitive)
        result=ProjectedSelectionIndex(primitives,index.width,index.height,tile_size=index.tile_size)
        self._selection_index=result;self._webgpu_pick_index=result
        return result

    def query_point(self,x,y,*,selection_filter=None,radius=None,config=None,front_only=False):
        from dataclasses import replace
        from .webgpu.scene import stipple_visible
        self.make_current()
        from .webgpu.picking import PointQuery
        query=getattr(self,"_point_query",None)
        if query is None:query=self._point_query=PointQuery(self)
        policy=config or self._selection_config
        effective_radius=round(self.pixel_ratio*policy.click_radius_px) if radius is None else max(0,int(radius))
        hits=query.query(x,y,selection_filter or policy.filter,effective_radius)
        styles={entry["item"]:entry["appearance"] for entry in self._display_entries().values()}
        height=self.framebuffer_size()[1]
        kept=[]
        for hit in hits:
            style=styles.get(hit.item,{})
            if style.get("opacity",1)<=0:continue
            if style.get("stipple") and not stipple_visible(style["stipple"],style.get("stipple_phase",0),x,y,height):continue
            kept.append(hit)
        if kept and len(kept)!=len(hits):
            first=min(hit.depth for hit in kept)
            return tuple(replace(hit,visible=hit.depth==first) for hit in kept)
        return tuple(kept)

    def set_highlight(self, tags, fill=None, outline=None):
        from .core import parse_color
        if fill is not None and parse_color(fill) is None:
            raise ValueError("unsupported highlight color")
        return super().set_highlight(tags, fill=fill,outline=outline)


    def destroy(self):
        if self._closed:
            return
        self.make_current()
        self._closed = True
        errors = []
        def attempt(callback):
            try:
                callback()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        attempt(self._cancel_hover)
        attempt(self._update_scheduler.close)
        for timer in tuple(self._timers):
            attempt(lambda timer=timer: self.after_cancel(timer))
        if getattr(self,"_point_query",None) is not None:
            attempt(self._point_query.close)
            self._point_query=None
        if hasattr(self.surface,"release"):attempt(self.surface.release)
        attempt(self._webgpu.close)
        attempt(self._renderer.release)
        self._entries.clear()
        attempt(self.hide)
        attempt(self.deleteLater)
        if errors:
            raise RuntimeError("WebGPU host cleanup failures: " + "; ".join(errors))

