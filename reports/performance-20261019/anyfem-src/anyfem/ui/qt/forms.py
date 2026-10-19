"""Typed option forms backed by the public engineering API defaults."""
import dataclasses
import inspect
from PySide6.QtWidgets import QWidget,QFormLayout,QLineEdit,QCheckBox
from .editors import CommandEditor,title


class OptionForm(QWidget):
    def __init__(self,parameters,parent=None):
        super().__init__(parent);layout=QFormLayout(self);self.fields={}
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for name,annotation,default in parameters:
            if str(annotation).startswith("bool") or isinstance(default,bool):
                widget=QCheckBox();widget.setChecked(bool(default))
            else:widget=CommandEditor.make_field(default,str(annotation),name)
            self.fields[name]=(widget,str(annotation),default)
            layout.addRow(title(name),widget)

    @classmethod
    def record(cls,record,values=None,exclude=()):
        parameters=[]
        for field in dataclasses.fields(record):
            if not field.init or field.name in exclude:continue
            default=(values or {}).get(field.name,field.default)
            if default is dataclasses.MISSING and field.default_factory is not dataclasses.MISSING:default=field.default_factory()
            parameters.append((field.name,field.type,default))
        return cls(parameters)

    def values(self):
        result={}
        for name,(widget,annotation,default) in self.fields.items():
            if isinstance(widget,QCheckBox):result[name]=widget.isChecked()
            elif widget.text().strip():result[name]=CommandEditor.value(self,widget,annotation,default)
            elif default is inspect.Parameter.empty or default is dataclasses.MISSING:raise ValueError(f"{title(name)} is required")
        return result

    def set_values(self,values):
        for name,value in values.items():
            if name not in self.fields:continue
            widget=self.fields[name][0]
            if isinstance(widget,QCheckBox):widget.setChecked(bool(value))
            else:widget.setText("" if value is None else CommandEditor.encoded(value))
