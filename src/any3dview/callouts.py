"""Bounded deterministic visual label placement; engineering anchors are immutable."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CalloutPlacement:
    key: str
    anchor: tuple
    text: str
    rect: tuple
    font: tuple
    depth: object = None

def _overlap(a,b):
    return a[0]<b[2]+3 and a[2]+3>b[0] and a[1]<b[3]+3 and a[3]+3>b[1]

def _elide(text,font,measure,maximum):
    text=' '.join(str(text).split())
    if measure(text,font)[0]<=maximum:return text
    if measure('\u2026',font)[0]>maximum:return ''
    low,high=0,len(text)
    while low<high:
        middle=(low+high+1)//2
        if measure(text[:middle]+'\u2026',font)[0]<=maximum:low=middle
        else:high=middle-1
    return text[:low]+'\u2026'

def layout_callouts(items,width,height,measure,*,maximum_labels=64):
    """Return placements and omitted count in logical pixels.

    Each item supplies key, text, anchor, font and optional depth. Stable keys
    determine order; 32 candidate positions and 64 visible labels bound work.
    Full text remains with the application; only the display copy is elided.
    """
    width,height=float(width),float(height)
    ordered=sorted(items,key=lambda i:(str(i['key']),str(i['text']),tuple(i['anchor'])))
    placements=[]
    if width<24 or height<48:return (),len(ordered)
    for item in ordered[:maximum_labels]:
        anchor=tuple(float(v) for v in item['anchor'])
        if not all(math.isfinite(v) for v in anchor) or not (0<=anchor[0]<=width and 0<=anchor[1]<=height):continue
        font=item['font'];text=_elide(item['text'],font,measure,min(228.,width-20.))
        if not text:continue
        tw,th=measure(text,font);w,h=tw+12,th+8
        if h>height-36:continue
        selected=None
        for radius in (16.,36.,64.,96.):
            for dx,dy in ((1,-1),(-1,-1),(1,1),(-1,1),(0,-1),(1,0),(0,1),(-1,0)):
                x=anchor[0]+dx*radius-(w if dx<0 else w*.5 if dx==0 else 0)
                y=anchor[1]+dy*radius-(h if dy<0 else h*.5 if dy==0 else 0)
                x=max(4.,min(x,width-w-4.));y=max(4.,min(y,height-h-28.))
                rect=(x,y,x+w,y+h)
                if rect[0]<=anchor[0]<=rect[2] and rect[1]<=anchor[1]<=rect[3]:continue
                if any(_overlap(rect,p.rect) for p in placements):continue
                selected=rect;break
            if selected is not None:break
        if selected is not None:placements.append(CalloutPlacement(str(item['key']),anchor,text,selected,font,item.get('depth')))
    return tuple(placements),len(ordered)-len(placements)

def draw_callouts(hud,items,width,height):
    measure=getattr(hud,'text_size',lambda text,font:(len(str(text))*max(6,float(font[1]))*.95,max(6,float(font[1]))*1.6))
    placements,omitted=layout_callouts(items,width,height,measure)
    for p in placements:
        x0,y0,x1,y1=p.rect
        end=(max(x0,min(p.anchor[0],x1)),max(y0,min(p.anchor[1],y1)))
        hud.line(p.anchor,end,'#596475',width=1,depth=p.depth)
        hud.rectangle(p.rect,'#667085',width=1,fill='#fffdf4',fill_alpha=.96,depth=p.depth)
        hud.text((x0+6,y0+4),p.text,'#17212f',font=p.font,anchor='nw',depth=p.depth)
    if omitted:
        text=f'+{omitted} callouts';font=('Segoe UI',9,'bold')
        tw,th=measure(text,font)
        if tw+12<=width-8 and th+8<=height-8:
            rect=(width-tw-16,height-th-12,width-4,height-4)
            hud.rectangle(rect,'#667085',fill='#fffdf4',fill_alpha=.96)
            hud.text((rect[0]+6,rect[1]+4),text,'#17212f',font=font,anchor='nw')
    return placements,omitted
