"""Validated visual policies; source scalar values remain unchanged."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Contours:
    mode: str = 'Continuous'
    levels: tuple = ()
    zero_contour: bool = False
    threshold: tuple | None = None

    def __post_init__(self):
        if self.mode not in ('Continuous','Banded','Isolines'): raise ValueError('unknown contour mode')
        levels=tuple(float(v) for v in self.levels)
        if len(levels)>16 or not all(math.isfinite(v) for v in levels): raise ValueError('at most 16 finite contour levels are supported')
        if any(a>=b for a,b in zip(levels,levels[1:])): raise ValueError('contour levels must be strictly increasing')
        object.__setattr__(self,'levels',levels)
        if self.threshold is not None:
            direction,value=self.threshold
            if direction not in ('above','below') or not math.isfinite(float(value)): raise ValueError('threshold requires above/below and finite value')
            object.__setattr__(self,'threshold',(direction,float(value)))

def contour_uniforms(settings,bounds):
    settings=settings or Contours()
    low,high=bounds;span=high-low or 1.
    levels=settings.levels
    if not levels and settings.mode!='Continuous': levels=tuple(low+span*i/10 for i in range(1,10))
    packed=np.zeros(16)
    packed[:len(levels)]=[(v-low)/span for v in levels]
    control=[('Continuous','Banded','Isolines').index(settings.mode),len(levels),float(settings.zero_contour),(0.-low)/span]
    emphasis=[0.,0.,0.,0.]
    if settings.threshold is not None:
        direction,value=settings.threshold;emphasis[:2]=[1. if direction=='above' else -1.,(value-low)/span]
    if not np.isfinite([*control,*packed,*emphasis]).all() or np.max(np.abs([*control,*packed,*emphasis]))>np.finfo(np.float32).max:
        raise ValueError('contour settings cannot be represented for this scalar range')
    return control,packed,emphasis
