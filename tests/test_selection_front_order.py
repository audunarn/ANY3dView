"""Visibility ordering must match a direct, non-tiled front-to-back scan."""
import random

from any3dview import PickBinding, ProjectedPrimitive, ProjectedSelectionIndex, SelectionDepth, SelectionFilter
from any3dview.selection import _primitive_distance


class ReferenceIndex(ProjectedSelectionIndex):
    def _front_primitive_at(self, point, radius=.75, selection_filter=None):
        # Same geometric predicate, independent broad-phase/order traversal.
        rect=(point[0]-radius,point[1]-radius,point[0]+radius,point[1]+radius)
        if rect[2]<0 or rect[3]<0 or rect[0]>self.width or rect[1]>self.height:return None
        for i in sorted(range(len(self.primitives)),key=lambda i:(self.primitives[i].depth,-self.primitives[i].layer,-i)):
            p=self.primitives[i];x0,y0,x1,y1=p.bbox
            if point[0]<x0-radius or point[0]>x1+radius or point[1]<y0-radius or point[1]>y1+radius:continue
            if selection_filter is not None and p.shape!='polygon':
                if p.binding is None or not any(selection_filter.accepts(o) for o in p.binding.owners):continue
            if _primitive_distance(p,point)<=radius:return i
        return None


def test_cached_front_order_matches_scan_for_tiles_globals_filters_and_ties():
    rng=random.Random(421);primitives=[]
    for i in range(120):
        x,y=rng.uniform(-20,480),rng.uniform(-20,480)
        if i%3==0:
            points=((x,y),(x+rng.uniform(2,350),y+2));shape='segment';kind='geometry.edge'
        else:
            w,h=rng.uniform(1,350),rng.uniform(1,350)
            points=((x,y),(x+w,y),(x+w,y+h),(x,y+h));shape='polygon';kind='geometry.face'
        primitives.append(ProjectedPrimitive(i,shape,points,(float(i%4),)*len(points),
            None if i%11==0 else PickBinding.one(str(i),kind),layer=float(i%3)))
    fast=ProjectedSelectionIndex(primitives,512,512);slow=ReferenceIndex(primitives,512,512)
    assert fast._global and fast._tiles
    filters=(SelectionFilter(),SelectionFilter(kinds=frozenset({'geometry.edge'})),SelectionFilter(kinds=frozenset({'geometry.face'})))
    for filt in filters:
        for point in [(rng.uniform(-5,517),rng.uniform(-5,517)) for _ in range(100)]+[(0,0),(512,512),(513,200)]:
            assert fast._front_primitive_at(point,selection_filter=filt)==slow._front_primitive_at(point,selection_filter=filt)
        for crossing in (False,True):
            for rect in ((0,0,512,512),(50,90,320,410),(200,200,201,201)):
                assert fast.rectangle_hits(rect,filt,crossing=crossing,depth=SelectionDepth.VISIBLE)==slow.rectangle_hits(rect,filt,crossing=crossing,depth=SelectionDepth.VISIBLE)
        assert fast.polygon_hits(((0,0),(480,40),(350,470)),filt,depth=SelectionDepth.VISIBLE)==slow.polygon_hits(((0,0),(480,40),(350,470)),filt,depth=SelectionDepth.VISIBLE)
