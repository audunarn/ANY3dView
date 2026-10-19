"""Descriptive, window-scoped help for existing and dynamic Qt buttons."""
from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import (QApplication, QAbstractButton, QCheckBox,
                              QRadioButton, QComboBox, QToolButton)


DESCRIPTIONS = {
    'New': 'Create a new project; you will be prompted about unsaved changes.',
    'Open': 'Open an existing ANYfem project.',
    'Save': 'Save the current project to its project file.',
    'Undo': 'Undo the most recent model edit.',
    'Redo': 'Restore the most recently undone model edit.',
    'Fit': 'Fit the visible model into the viewport.',
    'Point': 'Select points in the viewport.',
    'Line': 'Select lines in the viewport.',
    'Face': 'Select faces in the viewport.',
    'Generate mesh': 'Generate a mesh using the current mesh settings and project geometry.',
    'Solve': 'Start the selected analysis using the current mesh and analysis settings.',
    'Cancel mesh': 'Request cancellation of the active mesh job.',
    'Cancel solve': 'Request cancellation of the active analysis job.',
    'Cancel run': 'Request cancellation of the run selected in this monitor. Wait for the worker to acknowledge it.',
    'Copy displayed log': 'Copy the timestamped log displayed for the selected run to the clipboard.',
    'Follow active run': 'Automatically select a newly started run in the monitor.',
    'Pause graph': 'Pause graph updates while text and elapsed time continue updating.',
    'Run script': 'Run the editor script against a project snapshot and commit successful changes as one undoable edit.',
    'Cancel script': 'Request cancellation of the running script.',
    'Copy commands to editor': 'Copy the recorded GUI commands into the script editor for review or replay.',
    'Clear commands': 'Clear the recorded GUI command list.',
    'Clear output': 'Clear the displayed script output.',
    'Copy diagnosis': 'Copy script diagnostic information to the clipboard.',
    'Record GUI commands': 'Record successful GUI model commands for review or replay in the script editor.',
    'Create set from selection': 'Create a named geometry set containing the currently selected entities.',
    'Create region from selection': 'Create a reusable region from the current selection and chosen region settings.',
    'Create Boolean region': 'Create a region using the selected Boolean operation and operand regions.',
    'Apply unit profile': 'Apply the chosen project unit profile; stored engineering quantities retain their SI values.',
    'Create DNV material': 'Create a material using the chosen DNV material settings.',
    'Create plate section with DNV material': 'Create a plate section and its DNV material using the entered settings.',
    'Browse section library': 'Browse standard beam sections (T-bar, angle, bulb flat, flat bar) and load one into the beam section form for review.',
    'Apply support preset to selection': 'Assign the selected support preset to the selected geometry.',
    'Open file': 'Choose a file to inspect and display its import information.',
    'Save JSON report': 'Save the displayed file inspection report as JSON.',
    'Write canonical FEM': 'Write the inspected model in canonical FEM format.',
    'Check selected plates for mapped meshing': 'Check whether the selected plates support mapped meshing and show the diagnosis.',
    'Pin selected lines': 'Pin the selected lines to the current mesh seeding settings.',
    'Clear seeding pins': 'Remove the explicit mesh seeding pins.',
    'Refine around selection': 'Add local mesh refinement around the selected geometry.',
    'Clear refinements': 'Remove the configured local mesh refinements.',
    'Preview layout': 'Preview the mesh layout using the current settings before committing it.',
    'Commit layout preview': 'Commit the current mesh layout preview.',
    'Discard layout preview': 'Discard the mesh layout preview.',
    'Copy mesh diagnosis': 'Copy the current mesh diagnosis to the clipboard.',
    'Open standalone ANYmesher': 'Open the standalone mesher for additional mesh controls.',
    'Show results': 'Display the selected result field, component and frame in the viewport.',
    'Play retained frames': 'Animate the available result frames at the chosen playback rate.',
    'Stop playback': 'Stop the result animation.',
    'Inspect quantity / table': 'Inspect the selected result quantity or its numeric table.',
    'Export quantity field CSV': 'Export the selected result quantity as a CSV table.',
    'Probe selection': 'Inspect result values at the selected entities.',
    'Sample along selected line': 'Sample result values along the selected line.',
    'Plot history': 'Plot the selected result history.',
    'Plot at selected point': 'Plot result history at the selected point.',
    'Export field CSV': 'Export the current result field as CSV.',
    'Export report': 'Export a report for the selected results.',
    'Export GIF': 'Export the result animation as a GIF.',
    'Choose colour': 'Open the colour picker for this appearance setting.',
    'Apply appearance': 'Apply the entered viewport appearance settings.',
    'Reset appearance': 'Restore the default viewport appearance settings.',
    'Load viewport appearance': 'Load the current viewport appearance into these controls.',
    'Enable section plane': 'Clip the displayed model with the entered section plane.',
    'Clear section plane': 'Remove the viewport section plane.',
    'Save viewport image': 'Save an image of the current viewport.',
    'Start construction': 'Start interactive construction using the chosen tool and workplane.',
    'Construct plate from existing vertices': 'Activate an ordered boundary draft. Click existing vertices, review their order, then Enter or double-click to create boundary lines and one geometry face. Assign material and thickness separately.',
    'Apply plate': 'Validate the ordered boundary with ANYgeometry and create one geometry face as one undoable command.',
    'Sketch selected flat plate': 'Start a sketch on the selected flat plate.',
    'Add local point': 'Add a construction point at the entered local coordinates.',
    'Remove last point': 'Remove the most recently added construction point.',
    'Add distance constraint': 'Constrain the entered point pair to the specified distance.',
    'Add coincidence constraint': 'Constrain the entered point pair to coincide.',
    'Remove last constraint': 'Remove the most recently added sketch constraint.',
    'Copy': 'Copy the selected geometry by the entered displacement vector.',
    'Move': 'Move the selected geometry by the entered displacement vector.',
    'OK': 'Accept the entered values and close this dialog.',
    'Cancel': 'Cancel this operation and close its dialog or construction controls.',
    'Close': 'Close this window or dialog.',
}

WORKSPACES = {
    'Geometry': 'Create and edit points, lines, plates and geometry features.',
    'Construction': 'Construct geometry interactively with workplanes, sketches and constraints.',
    'Generators': 'Create editable structural features from parameterized generators.',
    'Materials': 'Define materials and their engineering properties.',
    'Sections': 'Define plate and beam sections.',
    'Loads BC': 'Define loads, constraints and boundary conditions.',
    'Mesh': 'Configure, preview and generate the finite element mesh.',
    'Solve': 'Configure and run analyses.',
    'Results': 'Inspect result fields, components, histories and animation.',
    'Visualization': 'Configure viewport appearance, clipping and image export.',
    'Scripts': 'Edit and run scripts and review recorded GUI commands.',
    'Definitions': 'Define regions, coordinate systems, units and output requests.',
}


def clean(text):
    return ' '.join(str(text or '').replace('&', '').replace('…', '').rstrip('.').split())


def button_description(button):
    """Preserve custom hints; expand plain labels into action descriptions."""
    text = clean(button.text() or button.accessibleName())
    existing = button.toolTip()
    generated = button.property('_anyfem_help')
    if existing and existing != generated and clean(existing) not in {text, text+' workspace'}:
        return existing
    if text == 'Apply':
        parent = button.parentWidget()
        while parent is not None:
            choice = getattr(parent, 'choice', None)
            if isinstance(choice, QComboBox):
                return f'Apply {choice.currentText()} using the entered values as an undoable model edit.'
            if type(parent).__name__ == 'ConstructionPanel':
                return 'Apply the current construction or sketch to the model.'
            parent = parent.parentWidget()
        return 'Apply the values entered in these controls.'
    if text == 'Create editable feature':
        parent = button.parentWidget()
        while parent is not None:
            choice = getattr(parent, 'kind', None)
            if isinstance(choice, QComboBox):
                return f'Create an editable {choice.currentText().lower()} feature using the entered parameters.'
            parent = parent.parentWidget()
    if text in DESCRIPTIONS:
        return DESCRIPTIONS[text]
    if isinstance(button, QToolButton) and button.defaultAction() is not None:
        action = button.defaultAction()
        text = clean(action.text())
        if text in WORKSPACES and action.isCheckable():
            return f'Open the {text} workspace. {WORKSPACES[text]}'
        return DESCRIPTIONS.get(text, action.toolTip() or f'Activate {text}.')
    if isinstance(button, (QCheckBox, QRadioButton)) and text:
        return f'Enable or disable {text.lower()}.' if isinstance(button, QCheckBox) else f'Choose {text.lower()}.'
    return existing or (f'Activate {text}.' if text else 'Open the available controls for this item.')


class ButtonHelp(QObject):
    """Covers dynamic dialogs without intercepting clicks or worker events."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.application = QApplication.instance()
        self.application.installEventFilter(self)
        self.refresh(window)

    def owns(self, widget):
        parent = widget
        while parent is not None:
            if parent is self.window:
                return True
            parent = parent.parentWidget()
        return False

    def describe(self, button):
        description = button_description(button)
        button.setToolTip(description)
        button.setProperty('_anyfem_help', description)

    def refresh(self, widget):
        if isinstance(widget, QAbstractButton):
            self.describe(widget)
        for button in widget.findChildren(QAbstractButton):
            self.describe(button)

    def eventFilter(self, target, event):
        if isinstance(target, QAbstractButton) and self.owns(target):
            if event.type() in (QEvent.ToolTip, QEvent.Show, QEvent.Enter):
                self.describe(target)
        return False

    def close(self):
        self.application.removeEventFilter(self)
