"""Searchable left-hand result worktree over the existing result contracts."""
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QSpinBox, QTreeView, QVBoxLayout, QWidget

from ...presentation.result_browser import result_entries
from ...presentation.result_display import unit_transform
from .navigation import navigation_icon


class ResultComponentSelector(QComboBox):
    """Editable component choices retaining the former text-control contract."""
    def __init__(self):
        super().__init__();self.setEditable(True)
    def text(self):return self.currentText()
    def setText(self,text):self.setCurrentText(text)
    def setPlaceholderText(self,text):self.lineEdit().setPlaceholderText(text)


class ResultFrameSelector(QSpinBox):
    """Present frame numbers from one while keeping existing zero-based APIs."""
    def textFromValue(self,value):return str(value+1)
    def valueFromText(self,text):return int(text)-1


class QtResultBrowser(QWidget):
    def __init__(self,app):
        super().__init__();self.app=app;self._signature=None;self._items={};self._updating=False
        layout=QVBoxLayout(self);layout.setContentsMargins(6,6,6,6)
        self.status=QLabel("Run or open an analysis to browse results.");self.status.setWordWrap(True)
        self.status.setToolTip("Only available result fields and components are listed. Saved fields retain their submitted model and units.")
        layout.addWidget(self.status)
        self.search=QLineEdit();self.search.setPlaceholderText("Find a result or component")
        self.search.setClearButtonEnabled(True);layout.addWidget(self.search)
        self.tree=QTreeView();self.tree.setUniformRowHeights(True);self.tree.setEditTriggers(QTreeView.NoEditTriggers)
        self.model=QStandardItemModel(self);self.model.setHorizontalHeaderLabels(["Results","Unit / location"])
        self.tree.setModel(self.model);self.tree.setColumnWidth(0,230);layout.addWidget(self.tree)
        self.search.textChanged.connect(self._filter)
        self.tree.selectionModel().currentChanged.connect(self._selected)

    def refresh(self):
        panel=self.app.panels["Results"];dataset=self.app.result_datasets.get(self.app.active_job_id)
        fields=panel.live_field_names() if self.app.solution is not None else ()
        entries=result_entries(live_fields=fields,live_quantities=panel._live_quantity_descriptors,dataset=dataset)
        owner=(self.app.project.document_id,self.app.active_job_id,id(self.app.solution),id(dataset))
        job=self.app.project.jobs.get(self.app.active_job_id)
        stale=job is not None and self.app._job_is_stale(job)
        name=job.name if job else "Active result"
        state=" · stale — submitted model" if stale else ""
        self.status.setText(f"{name}{state}\nFrame {panel.frame.value()+1} of {panel.frame.maximum()+1}" if entries else
                            "Run or open an analysis to browse results.")
        signature=(owner,entries,panel.display_units())
        if signature!=self._signature:
            self._updating=True
            expanded=self._expanded_paths() if self._signature and owner==self._signature[0] else None
            try:
                self.model.removeRows(0,self.model.rowCount());self._items={};parents={():self.model.invisibleRootItem()};quantity_parents={}
                for entry in entries:
                    quantity=(entry.kind in {"field","field_table","live"})
                    quantity_key=(entry.path[0],entry.key) if quantity else None
                    parent=quantity_parents.get(quantity_key) if quantity else None
                    path=()
                    for label in entry.path:
                        path+=(label,)
                        if parent is not None:continue
                        if path not in parents:
                            item=QStandardItem(label);item.setEditable(False);item.setData(path,Qt.UserRole+1)
                            font=item.font();font.setBold(len(path)<=2);item.setFont(font)
                            if len(path)<=2:item.setIcon(navigation_icon("Results"))
                            parents[path[:-1]].appendRow([item,QStandardItem()]);parents[path]=item
                    item=QStandardItem(entry.label);item.setEditable(False);item.setData(entry,Qt.UserRole)
                    item.setToolTip(f"{entry.key}\n{entry.location or 'Live scalar field'} · {entry.unit or 'dimensionless'}")
                    display_unit=unit_transform(entry.unit,panel.display_units())[1]
                    detail=" · ".join(value for value in (display_unit,entry.location) if value)
                    (parent if parent is not None else parents[path]).appendRow([item,QStandardItem(detail)]);self._items[entry.identity]=item
                    if quantity and quantity_key not in quantity_parents:
                        quantity_parents[quantity_key]=item
                        item.setData(("quantity",*quantity_key),Qt.UserRole+1)
                if expanded is None:self.tree.expandToDepth(1)
                else:
                    for path,item in parents.items():
                        if path in expanded:self.tree.setExpanded(item.index(),True)
                    for item in quantity_parents.values():
                        if tuple(item.data(Qt.UserRole+1)) in expanded:self.tree.setExpanded(item.index(),True)
                self._signature=signature;self._filter(self.search.text())
            finally:self._updating=False
        self.sync_selection()

    def _expanded_paths(self):
        paths=set()
        def visit(parent):
            for row in range(parent.rowCount()):
                item=parent.child(row)
                if self.tree.isExpanded(item.index()):paths.add(tuple(item.data(Qt.UserRole+1) or ()))
                visit(item)
        visit(self.model.invisibleRootItem());return paths

    def _filter(self,text):
        needle=text.strip().casefold()
        def visit(parent,ancestor_match=False):
            any_visible=False
            for row in range(parent.rowCount()):
                item=parent.child(row);matches=ancestor_match or needle in item.text().casefold()
                child_visible=visit(item,matches) if item.hasChildren() else False
                visible=not needle or matches or child_visible
                self.tree.setRowHidden(row,parent.index(),not visible);any_visible|=visible
                if needle and child_visible:self.tree.setExpanded(item.index(),True)
            return any_visible
        visit(self.model.invisibleRootItem())

    def _selected(self,current,previous):
        if self._updating:return
        entry=current.siblingAtColumn(0).data(Qt.UserRole)
        if entry is not None:self.app.guarded(lambda:self.app.panels["Results"].select_entry(entry))()

    def sync_selection(self):
        panel=self.app.panels["Results"]
        key=panel._selected_entry
        item=self._items.get(key)
        if item is not None:
            blocker=QSignalBlocker(self.tree.selectionModel())
            self.tree.setCurrentIndex(item.index())
            del blocker
