"""Non-mutating ordered existing-vertex draft; geometry truth stays with its owner."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from ... import commands


class PlateDraft(QDialog):
    def __init__(self, actions, refs=()):
        super().__init__(actions.app)
        self.actions = actions
        self.binding = actions.binding()
        self.ids = [ref.id for ref in refs]
        self.finished_command = False
        self.setWindowTitle("Add plate — boundary order")
        self.setObjectName("PlateBoundaryDraft")
        layout = QVBoxLayout(self)
        guide = QLabel("Review boundary order; box/lasso selection does not establish polygon order. "
                       "Edit comma-separated vertex IDs, or click existing vertices in the viewport. "
                       "Enter or double-click applies; Escape cancels. Material and thickness are assigned separately.")
        guide.setWordWrap(True);layout.addWidget(guide)
        self.order = QLineEdit(", ".join(map(str, self.ids)))
        self.order.setObjectName("PlateVertexOrder");layout.addWidget(self.order)
        self.summary = QLabel();layout.addWidget(self.summary)
        self.feedback = QLabel();self.feedback.setWordWrap(True);layout.addWidget(self.feedback)
        self.order.textEdited.connect(self.edit_order)
        for label, callback in (("Apply plate", self.commit), ("Cancel", self.reject)):
            button = QPushButton(label)
            button.setToolTip("Validate this boundary with ANYgeometry and create one undoable geometry face." if label == "Apply plate" else "Discard the uncommitted boundary preview.")
            button.clicked.connect(callback);layout.addWidget(button)
        self.finished.connect(self.cleanup)
        self.update_preview()

    def check_binding(self):
        self.actions.assert_binding(self.binding)
        if self.actions.app.session.read_only or self.actions.app.project.read_only_reason:
            raise PermissionError("This project is read-only")

    def edit_order(self, text):
        try:
            ids = [int(value.strip()) for value in text.split(",") if value.strip()]
            self.check_binding()
            for identifier in ids:self.actions.app.project.geometry.entity_ref("vertex", identifier)
            self.ids = ids
            self.feedback.clear();self.update_preview()
        except (ValueError, KeyError, PermissionError) as error:
            self.ids = []
            self.update_preview()
            self.summary.setText("Invalid vertex order — preview unavailable")
            self.feedback.setText(str(error))

    def add_vertex(self, ref):
        try:self.check_binding()
        except (ValueError, PermissionError) as error:
            self.feedback.setText(str(error));return
        if ref is None or ref.kind != "vertex":return
        if self.ids and self.ids[-1] == ref.id:return
        self.ids.append(ref.id)
        self.order.setText(", ".join(map(str, self.ids)));self.update_preview()

    def update_preview(self):
        self.summary.setText("Boundary preview: " + " → ".join(f"{i+1}: vertex {identifier}" for i, identifier in enumerate(self.ids)))
        viewport = self.actions.app.viewport
        if getattr(viewport, "plate_draft", None) is self and viewport._scene is not None:
            viewport._render_scene(viewport._scene)

    def commit(self):
        if self.finished_command:return
        try:
            self.check_binding()
            # Reparse even programmatic field edits; never apply a stale preview.
            ids = [int(value.strip()) for value in self.order.text().split(",") if value.strip()]
            if len(ids) < 3:raise ValueError("A plate needs at least three boundary vertices.")
            if len(set(ids)) != len(ids):raise ValueError("Each boundary vertex must occur once; omit the repeated closing vertex.")
            geometry = self.actions.app.project.geometry.clone()
            face = geometry.add_plate(ids)
            edges = {item.edge for item in geometry.faces[face].loop}
            prefixes = (f"face {face} ", *(f"edge {identifier} " for identifier in edges))
            errors = [error for error in geometry.validate_topology() if error.startswith(prefixes)]
            if errors:raise ValueError("Invalid plate boundary: " + "; ".join(errors))
            self.actions.app.run(commands.AddPlate(ids))
        except (ValueError, RuntimeError, KeyError, PermissionError) as error:
            self.feedback.setText(str(error));return
        self.finished_command = True
        self.accept()

    def cleanup(self, *_):
        viewport = self.actions.app.viewport
        if getattr(viewport, "plate_draft", None) is self:
            viewport.plate_draft = None
            if not viewport._closed and viewport._scene is not None:viewport._render_scene(viewport._scene)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):self.commit();event.accept()
        else:super().keyPressEvent(event)
