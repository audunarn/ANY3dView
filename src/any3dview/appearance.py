"""Immutable retained scalar appearance; generic display interpolation only."""
from dataclasses import dataclass
import math
import re
import numpy as np

_UNSET=object()
@dataclass(frozen=True,slots=True)
class MeshAppearance:
    scalar_range: tuple[float,float] | None=None
    colormap: tuple[tuple[float,str],...] | None=None

def validated_appearance(scalar_range=None,colormap=None):
    if scalar_range is not None:
        if len(scalar_range)!=2 or any(isinstance(v,(bool,np.bool_)) for v in scalar_range):raise ValueError('scalar_range requires two finite ordered values')
        scalar_range=tuple(float(v) for v in scalar_range)
        if not all(math.isfinite(v) for v in scalar_range) or scalar_range[0]>scalar_range[1]:raise ValueError('scalar_range requires two finite ordered values')
    if colormap is not None:
        stops=[]
        for position,color in colormap:
            if isinstance(position,(bool,np.bool_)):raise ValueError('colormap positions require finite0..1 values')
            position=float(position)
            if not math.isfinite(position) or not 0<=position<=1 or (stops and position<stops[-1][0]):raise ValueError('colormap positions must be ordered within0..1')
            if not isinstance(color,str) or not re.fullmatch(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?',color):raise ValueError('colormap colors must be RGB hex')
            color=color.lower()
            if len(color)==4:color='#'+''.join(c*2 for c in color[1:])
            stops.append((position,color))
        if not stops:raise ValueError('colormap requires at least one stop')
        colormap=tuple(stops)
    return MeshAppearance(scalar_range,colormap)

def scalar_color(value,minimum,maximum,colormap=None):
    if colormap is None:
        from .core import _interpolate_thickness_color
        return _interpolate_thickness_color(value,minimum,maximum)
    t=min(1.,max(0.,(float(value)-minimum)/max(1e-20,maximum-minimum)))
    positions=[p for p,_ in colormap]
    colors=np.array([[int(c[j:j+2],16) for j in (1,3,5)] for _,c in colormap],dtype=float)
    rgb=[int(round(float(np.interp(t,positions,colors[:,i])))) for i in range(3)]
    return '#'+''.join(f'{c:02x}' for c in rgb)

def palette_data(colormap=None):
    # Same fixed256-entry LUT as the existing renderer, retainingendpoint clamp.
    colors=[scalar_color(i/255.,0.,1.,colormap) for i in range(256)]
    return np.asarray([[int(c[j:j+2],16)/255. for j in (1,3,5)] for c in colors],dtype=np.float32).tobytes()
