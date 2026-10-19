"""Read-only Qt model for engineering table previews."""
from PySide6.QtCore import QAbstractTableModel,Qt
from PySide6.QtWidgets import QTableView


class ResultTableModel(QAbstractTableModel):
    def __init__(self,parent=None):
        super().__init__(parent);self.headers=[];self.rows=[]

    def replace(self,headers,rows):
        self.beginResetModel();self.headers=list(headers);self.rows=list(rows);self.endResetModel()

    def rowCount(self,parent=None):return len(self.rows)
    def columnCount(self,parent=None):return len(self.headers)
    def data(self,index,role=Qt.DisplayRole):
        if role==Qt.DisplayRole and index.isValid():return str(self.rows[index.row()][index.column()])
    def headerData(self,section,orientation,role=Qt.DisplayRole):
        if role==Qt.DisplayRole:return self.headers[section] if orientation==Qt.Horizontal else str(section+1)


class ResultTable(QTableView):
    def __init__(self,parent=None):
        super().__init__(parent);self.values=ResultTableModel(self);self.setModel(self.values);self.setMinimumHeight(160)
    def show_rows(self,headers,rows):
        self.values.replace(headers,rows);self.resizeColumnsToContents()
