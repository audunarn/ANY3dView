"""Optional PySide6 frontend; importing ANYfem itself never loads Qt."""
def main():
    from .app import main as launch
    return launch()

def __getattr__(name):
    if name == "QtFemWindow":
        from .app import QtFemWindow
        return QtFemWindow
    raise AttributeError(name)
