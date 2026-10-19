"""Readable Qt engineering histories; missing samples remain visible gaps."""
import numpy as np
from PySide6.QtCore import QPointF, Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QToolTip


def label(text,unit):
    return f'{text} ({unit})' if unit else text


class HistoryPlot(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.series=[];self.empty_text='No history data'
        self.y_limits=None
        self._samples=[];self.setMinimumHeight(190);self.setMinimumWidth(260)
        self.setMouseTracking(True)

    def show_series(self,series):
        self.series=list(series);self._samples=[];self.update()

    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor('#ffffff'))
        painter.setRenderHint(QPainter.Antialiasing);self._samples=[]
        curves=[]
        for item in self.series:
            finite=np.isfinite(item.x)&np.isfinite(item.y)
            if finite.any():curves.append((item,finite))
        if not curves:
            painter.setPen(QColor('#64748b'))
            painter.drawText(self.rect().adjusted(12,12,-12,-12),Qt.AlignCenter|Qt.TextWordWrap,self.empty_text)
            return
        xmin=min(float(item.x[finite].min()) for item,finite in curves)
        xmax=max(float(item.x[finite].max()) for item,finite in curves)
        ymin=min(float(item.y[finite].min()) for item,finite in curves)
        ymax=max(float(item.y[finite].max()) for item,finite in curves)
        xpad=(xmax-xmin)*.04 or max(abs(xmin)*.05,1.)
        ypad=(ymax-ymin)*.08 or max(abs(ymin)*.05,1.)
        xmin-=xpad;xmax+=xpad;ymin-=ypad;ymax+=ypad
        if self.y_limits is not None:ymin,ymax=self.y_limits
        left,top,right,bottom=76,30+18*((len(curves)-1)//2),self.width()-20,self.height()-48
        if right<=left or bottom<=top:return
        def project(x,y):return QPointF(left+(float(x)-xmin)/(xmax-xmin)*(right-left),bottom-(float(y)-ymin)/(ymax-ymin)*(bottom-top))
        for fraction in np.linspace(0,1,5):
            x=left+fraction*(right-left);y=bottom-fraction*(bottom-top)
            painter.setPen(QPen(QColor('#e8edf4'),1))
            painter.drawLine(QPointF(x,top),QPointF(x,bottom));painter.drawLine(QPointF(left,y),QPointF(right,y))
            painter.setPen(QColor('#64748b'))
            painter.drawText(QRectF(x-30,bottom+4,60,18),Qt.AlignCenter,f'{xmin+fraction*(xmax-xmin):.3g}')
            painter.drawText(QRectF(2,y-9,66,18),Qt.AlignRight|Qt.AlignVCenter,f'{ymin+fraction*(ymax-ymin):.3g}')
        painter.setPen(QPen(QColor('#94a3b8'),1));painter.drawLine(left,bottom,right,bottom);painter.drawLine(left,top,left,bottom)
        for index,(item,finite) in enumerate(curves):
            color=QColor(['#2563eb','#dc2626','#059669','#7c3aed'][index%4])
            painter.setPen(QPen(color,1.8))
            # Draw each contiguous finite run separately; never bridge absent data.
            indices=np.flatnonzero(finite)
            for chunk in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                points=[project(item.x[i],item.y[i]) for i in chunk]
                if len(points)>1:painter.drawPolyline(QPolygonF(points))
                elif points:painter.drawEllipse(points[0],2.5,2.5)
            last=indices[-1];point=project(item.x[last],item.y[last])
            painter.setBrush(color);painter.drawEllipse(point,3.5,3.5);painter.setBrush(Qt.NoBrush)
            column=index%2;row=index//2;width=(right-left)/(1 if len(curves)==1 else 2)
            legend=f'{item.name} · {item.y[last]:.4g}'
            painter.drawText(QRectF(left+column*width,4+row*18,width-10,18),Qt.AlignLeft|Qt.AlignVCenter,
                             painter.fontMetrics().elidedText(legend,Qt.ElideRight,int(width-10)))
            self._samples.extend((project(item.x[i],item.y[i]),item,i) for i in indices)
        painter.setPen(QColor('#475569'))
        first=curves[0][0]
        painter.drawText(QRectF(left,bottom+25,right-left,20),Qt.AlignCenter,label(first.x_label,first.x_unit))
        painter.save();painter.translate(14,(top+bottom)/2);painter.rotate(-90)
        painter.drawText(QRectF(-(bottom-top)/2,-10,bottom-top,18),Qt.AlignCenter,label(first.y_label,first.y_unit));painter.restore()

    def mouseMoveEvent(self,event):
        if not self._samples:return
        position=event.position()
        point,item,index=min(self._samples,key=lambda row:(row[0].x()-position.x())**2+(row[0].y()-position.y())**2)
        if (point.x()-position.x())**2+(point.y()-position.y())**2>144:
            QToolTip.hideText();return
        text=f'{item.name}\n{label(item.x_label,item.x_unit)}: {item.x[index]:.6g}\n{label(item.y_label,item.y_unit)}: {item.y[index]:.6g}'
        QToolTip.showText(event.globalPosition().toPoint(),text,self)

    def leaveEvent(self,event):
        QToolTip.hideText();super().leaveEvent(event)
