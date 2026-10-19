"""Qt model/view representation of project records and persistent entity references."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor, QBrush
from PySide6.QtWidgets import QTreeView,QHeaderView,QMenu,QLineEdit,QVBoxLayout,QWidget,QPushButton
from anygeometry import feature_entity_owners
from ...presentation.tree import bounded_entity_ids, bounded_unowned_entity_ids
from ...presentation.points import displayed_geometry_point_ids
from .geometry_actions import is_geometry_set,set_references


LOAD_CASE_CHILD_LIMIT=200  # steps listed under a series; the rest are summarised

class QtModelTree(QTreeView):
    def __init__(self, app):
        super().__init__(app)
        self.app=app
        self.project=app.project
        self.exploded_feature_ids=frozenset()
        self.query=""
        self._model=QStandardItemModel()
        self._model.setHorizontalHeaderLabels(["Model", "State"])
        self.setModel(self._model)
        self.header().setSectionResizeMode(0,QHeaderView.Stretch)
        self.header().setSectionResizeMode(1,QHeaderView.Interactive)
        self.header().resizeSection(1,55)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.selectionModel().selectionChanged.connect(self._picked)
        self.doubleClicked.connect(self._activated)
        self._refreshing=False
        self._groups={}
        self._rows={}
        self._ref_rows={}
        self._beam_refs={}
        self._ref_beams={}
        self._visibility_rows={}
        self._visibility_keys={}
        self._hidden_keys=frozenset()
        self._row_order={}
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def beam_references(self, identifier):
        geometry = self.project.geometry
        member = geometry.members.get(identifier)
        if member is None:return ()
        return tuple(geometry.entity_ref("edge", geometry.member_edge_uses[use].edge_id)
                     for use in member.edge_use_ids)

    def generated_entity_owners(self):
        return feature_entity_owners(self.project.geometry)

    def reset_feature_exposure(self):
        self.exploded_feature_ids=frozenset()
        self.app.viewport.reset_visibility()
        self.app.viewport.set_collection_highlight(())
        self.app.viewport.set_feature_highlight(())
        actions=getattr(self.app,"geometry_actions",None)
        if actions is not None:actions.close()

    def dismiss_feature_selection(self):
        from PySide6.QtCore import QItemSelectionModel
        prior=self._refreshing;self._refreshing=True
        try:
            for index in self.selectionModel().selectedRows():
                value=index.data(Qt.UserRole)
                if isinstance(value,(tuple,list)) and value[0] in {"geometry","sets","wet_surfaces","connections","members"}:
                    self.selectionModel().select(index,QItemSelectionModel.Deselect|QItemSelectionModel.Rows)
        finally:self._refreshing=prior

    def search(self,text):
        self.query=text
        self.refresh()

    def panel(self):
        panel=QWidget(self.app);layout=QVBoxLayout(panel)
        search=QLineEdit(panel);search.setPlaceholderText("Find topology by entity ID")
        search.textChanged.connect(self.search)
        layout.addWidget(search);layout.addWidget(self)
        create=QPushButton("Create set from selection…",panel)
        create.clicked.connect(lambda:self.app.guarded(self.app.geometry_actions.new_set)())
        layout.addWidget(create)
        return panel

    def refresh(self):
        self._refreshing=True
        try:
            self.project=self.app.project
            geometry=self.project.geometry
            owners=self.generated_entity_owners()
            owned={kind:set() for kind in ("vertex","edge","face")}
            generated={}
            for ref,feature_id in owners.items():
                owned[ref.kind].add(ref.id)
                generated.setdefault(feature_id,{}).setdefault(ref.kind,[]).append(ref.id)
            member_edges = {use.edge_id for use in geometry.member_edge_uses.values()}
            owned["edge"].update(member_edges)
            self._sync_group("Beams", [(identifier, geometry.members[identifier].name or f"Beam {identifier}",
                                       f"{len(geometry.members[identifier].edge_use_ids)} segments", ("members",identifier))
                                      for identifier in bounded_entity_ids(geometry.members,self.query)])
            for label,kind,items in [("Points","vertex",displayed_geometry_point_ids(geometry)),("Lines","edge",geometry.edges),("Plates","face",geometry.faces)]:
                self._sync_group(label,[(identifier,f"{kind.capitalize()} {identifier}","",geometry.entity_ref(kind,identifier)) for identifier in bounded_unowned_entity_ids(items,owned[kind],self.query)])
            sets=[region for region in self.project.regions if is_geometry_set(region)]
            contents={}
            unresolved=set()
            for region in sets:
                try:contents[region.id]=set_references(self.project,region.id)
                except (ValueError,KeyError):contents[region.id]=();unresolved.add(region.id)
            self._sync_group("Sets",[(region.id,region.name,"unresolved" if region.id in unresolved else str(len(contents[region.id])),("sets",region.id)) for region in sets])
            for identifier,(item,_) in self._rows["Sets"].items():
                item.removeRows(0,item.rowCount())
                refs=contents[identifier]
                for kind in ("vertex","edge","face"):
                    by_id={ref.id:ref for ref in refs if ref.kind==kind}
                    for key in bounded_entity_ids(sorted(by_id),self.query):
                        child=QStandardItem(f"{kind.capitalize()} {key}");child.setEditable(False)
                        child.setData(by_id[key],Qt.UserRole);item.appendRow(child)
            wet=list(self.project.wet_surfaces)
            wet_contents={}
            for surface in wet:
                try:wet_contents[surface.id]=self.wet_surface_references(surface)
                except (ValueError,KeyError):wet_contents[surface.id]=None
            self._sync_group("Wet surfaces",[(surface.id,surface.name,"unresolved" if wet_contents[surface.id] is None else str(len(wet_contents[surface.id])),("wet_surfaces",surface.id)) for surface in wet])
            for identifier,(item,_) in self._rows["Wet surfaces"].items():
                item.removeRows(0,item.rowCount())
                for ref in sorted(wet_contents.get(identifier) or (),key=lambda ref:(ref.kind,ref.id)):
                    if self.query and self.query not in str(ref.id):continue
                    child=QStandardItem(f"{ref.kind.capitalize()} {ref.id}");child.setEditable(False)
                    child.setData(ref,Qt.UserRole);item.appendRow(child)
            for label,attribute in [("Features","geometry"),("Materials","materials"),("Plate sections","plate_sections"),("Beam sections","beam_sections"),("Load cases","load_cases"),("Supports","supports"),("Masses","masses"),("Connections","connections"),("Imperfections","imperfections"),("Coordinate systems","coordinate_systems"),("Regions","regions"),("Output requests","output_requests"),("Meshes","mesh_records"),("Analyses","analyses"),("Jobs","jobs")]:
                values=getattr(self.project,attribute,{})
                if attribute=="geometry":
                    history=getattr(geometry,"features",None)
                    values={} if history is None else {record.feature_id:record for record in history.records}
                pairs=values.items() if hasattr(values,"items") else enumerate(values)
                rows=[];children_of={}
                for key,value in pairs:
                    if attribute=="regions" and is_geometry_set(value):continue
                    if attribute=="load_cases":
                        # Steps hang under their series definition (or, for cases
                        # generated by copying, under a group named by the series).
                        if value.parent is not None:continue
                        if value.series is not None and value.times is None:
                            children_of.setdefault(f"series:{value.series}",[]).append(value);continue
                    key=getattr(value,"id",key)
                    if attribute=="materials":key=self.project.material_ids.get(value.name,key)
                    state=getattr(value,"status","")
                    status=str(getattr(state,"value",state))
                    if attribute=="jobs" and self.app._job_is_stale(value):
                        status += " (stale)"
                    if attribute=="mesh_records" and value.source_model_hash!=self.app.session.revision.model_hash:
                        status += " (stale)"
                    if attribute=="load_cases" and value.times is not None:
                        children_of[key]=self.project.series_children(value.name)
                        status=f"series, {len(children_of[key])} steps"
                    rows.append((key,str(getattr(value,"name",key)),status,(attribute,key)))
                if attribute=="load_cases":
                    for group,cases in children_of.items():
                        if group.startswith("series:"):
                            rows.append((group,group.split(":",1)[1],f"series, {len(cases)} cases",("series",group.split(":",1)[1])))
                self._sync_group(label,rows)
                if attribute=="load_cases":
                    for group,cases in children_of.items():
                        item=self._rows[label][group][0];item.removeRows(0,item.rowCount())
                        for case in cases[:LOAD_CASE_CHILD_LIMIT]:
                            child=QStandardItem(case.name);child.setEditable(False)
                            child.setData(("load_cases",case.id),Qt.UserRole);item.appendRow(child)
                        if len(cases)>LOAD_CASE_CHILD_LIMIT:
                            more=QStandardItem(f"... {len(cases)-LOAD_CASE_CHILD_LIMIT} more");more.setEditable(False);item.appendRow(more)
                if attribute=="geometry":
                    for feature_id,(item,_) in self._rows[label].items():
                        item.removeRows(0,item.rowCount())
                        if feature_id not in self.exploded_feature_ids:continue
                        for kind,identifiers in generated.get(feature_id,{}).items():
                            branch=QStandardItem(f"Generated {kind}s ({len(identifiers)})");branch.setEditable(False)
                            item.appendRow(branch)
                            for identifier in bounded_entity_ids(sorted(identifiers),self.query):
                                child=QStandardItem(f"{kind.capitalize()} {identifier}");child.setEditable(False)
                                child.setData(geometry.entity_ref(kind,identifier),Qt.UserRole);branch.appendRow(child)
            loads=[]
            for case in self.project.load_cases.values():
                for collection in ("point_loads","pressures","line_loads","surface_tractions","wet_pressures","connection_loads"):
                    for value in getattr(case,collection,()):
                        label=f"{case.name}: Wet pressure ({value.kind})" if collection=="wet_pressures" else f"{case.name}: Connection load ({value.name})" if collection=="connection_loads" else f"{case.name}: {type(value).__name__}"
                        loads.append((value.id,label,"",("load",value.id)))
            self._sync_group("Loads",loads)
            self._sync_group("Results",[(job.result_artifact_id,job.name,"stale" if self.app._job_is_stale(job) else "",("result",job.result_artifact_id)) for job in self.project.jobs.values() if job.result_artifact_id])
            self._index_rows(owners)
            self.refresh_visibility(rebuild=True)
            self.sync_from_selection()
        finally:
            self._refreshing=False
        selected=[index.data(Qt.UserRole) for index in self.selectionModel().selectedRows()]
        collection=[]
        for value in selected:
            try:collection.extend(self.collection_references(value))
            except (ValueError,KeyError):pass
        for value in selected:
            if isinstance(value,(tuple,list)) and value[0]=="members":collection.extend(self.beam_references(value[1]))
        self.app.viewport.set_collection_highlight(collection)

    def wet_surface_references(self,surface):
        """The plates a wet surface covers, resolved like a set's members."""
        if surface.region is not None:return set_references(self.project,surface.region.id)
        return tuple(self.project.geometry.resolve_ref(surface.ref))

    def collection_references(self,value):
        """Geometry behind a set or wet-surface row, for highlight and hiding."""
        if not isinstance(value,(tuple,list)):return ()
        if value[0]=="sets":return set_references(self.project,value[1])
        if value[0]=="wet_surfaces":
            surface=next((item for item in self.project.wet_surfaces if item.id==value[1]),None)
            return () if surface is None else self.wet_surface_references(surface)
        if value[0]=="connections":
            connection=next((item for item in self.project.connections if item.id==value[1]),None)
            return () if connection is None else tuple(self.project.connection_targets(connection))
        return ()

    def _index_rows(self,owners):
        """Rebuild indexes after structural edits, including duplicate refs."""
        self._ref_rows={};self._beam_refs={};self._ref_beams={}
        self._visibility_rows={};self._visibility_keys={}
        self._row_order={}
        feature_refs={}
        for ref,identifier in owners.items():feature_refs.setdefault(identifier,[]).append(ref)
        def visit(parent):
            for row in range(parent.rowCount()):
                item=parent.child(row);value=item.data(Qt.UserRole);refs=()
                self._row_order[item.index()]=len(self._row_order)
                if hasattr(value,"kind"):
                    refs=(value,)
                    self._ref_rows.setdefault(value,[]).append(item)
                elif isinstance(value,(tuple,list)):
                    if value[0]=="geometry":refs=feature_refs.get(value[1],())
                    elif value[0]=="members":
                        refs=self.beam_references(value[1])
                        self._beam_refs[value[1]]=frozenset(refs)
                        for ref in refs:self._ref_beams.setdefault(ref,set()).add(value[1])
                    elif value[0] in {"sets","wet_surfaces","connections"}:
                        try:refs=self.collection_references(value)
                        except (ValueError,KeyError):pass
                keys=frozenset(self.app.viewport.visibility.key(ref) for ref in refs)
                if keys or item.toolTip():
                    token=id(item)
                    self._visibility_rows[token]=(item,keys)
                    for key in keys:self._visibility_keys.setdefault(key,set()).add(token)
                visit(item)
        visit(self._model.invisibleRootItem())

    def refresh_visibility(self,*,rebuild=False):
        """Update only rows affected by visibility changes, in every occurrence."""
        hidden=frozenset(self.app.viewport.visibility.hidden)
        if rebuild:affected=self._visibility_rows
        else:
            affected=set()
            for key in hidden ^ self._hidden_keys:
                affected.update(self._visibility_keys.get(key,()))
        for token in affected:
            item,keys=self._visibility_rows[token]
            count=len(keys & hidden)
            fully_hidden=bool(keys) and count==len(keys)
            tooltip="Hidden in viewport" if fully_hidden else "Partly hidden in viewport" if count else ""
            brush=QBrush(QColor("#8794a6")) if fully_hidden else QBrush()
            if item.foreground()!=brush:item.setForeground(brush)
            if item.toolTip()!=tooltip:item.setToolTip(tooltip)
        self._hidden_keys=hidden

    def _sync_group(self,label,rows):
        parent=self._groups.get(label)
        if parent is None:
            parent=QStandardItem(label);parent.setEditable(False)
            self._model.appendRow(parent);self._groups[label]=parent
            self._rows[label]={}
        existing=self._rows[label];wanted={key for key,*_ in rows}
        for key in tuple(existing):
            if key not in wanted:
                parent.removeRow(existing[key][0].row());existing.pop(key)
        for key,text,state,data in rows:
            pair=existing.get(key)
            if pair is None:
                item=QStandardItem(text);item.setEditable(False)
                badge=QStandardItem(state);badge.setEditable(False)
                item.setData(data,Qt.UserRole);parent.appendRow([item,badge])
                existing[key]=(item,badge)
            else:
                item,badge=pair
                if item.text()!=text:item.setText(text)
                if badge.text()!=state:badge.setText(state)
                if item.data(Qt.UserRole)!=data:item.setData(data,Qt.UserRole)

    def _picked(self,*_):
        if self._refreshing:
            return
        refs=[index.data(Qt.UserRole) for index in self.selectionModel().selectedRows()]
        features=[ref[1] for ref in refs if isinstance(ref,(tuple,list)) and ref[0]=="geometry"]
        collections=[]
        for value in refs:
            if isinstance(value,(tuple,list)) and value[0] in {"sets","wet_surfaces","connections"}:
                try:collections.extend(self.collection_references(value))
                except (ValueError,KeyError) as exc:self.app.set_status(str(exc),error=True)
        for value in refs:
            if isinstance(value,(tuple,list)) and value[0]=="members":collections.extend(self.beam_references(value[1]))
        refs=[ref for ref in refs if hasattr(ref,"kind") and hasattr(ref,"id")]
        refs=list(dict.fromkeys(refs+collections))
        self._tree_driven=True
        try:
            if refs and len({ref.kind for ref in refs})==1:self.app.selection.set_mode(refs[0].kind)
            self.app.selection.clear()
            for ref in refs:
                if ref.kind==self.app.selection.mode:self.app.selection.select(ref,extend=True)
            self.app.viewport.set_feature_highlight(features)
            self.app.viewport.set_collection_highlight(collections)
        finally:self._tree_driven=False

    def _activated(self,index):
        value=index.data(Qt.UserRole)
        if isinstance(value,(tuple,list)) and value[0]=="jobs":
            self.app.guarded(lambda:self.app.panels["Results"].activate_job(value[1]))()
        elif isinstance(value,(tuple,list)) and value[0]=="result":
            job=next(job for job in self.project.jobs.values() if job.result_artifact_id==value[1])
            self.app.guarded(lambda:self.app.panels["Results"].activate_job(job.id))()
        else:self.app.guarded(lambda:self.edit_record(value))()

    @staticmethod
    def item_key(value):
        if hasattr(value,"kind"):return f"{value.kind}:{value.id}"
        if not isinstance(value,(tuple,list)):return None
        attribute,key=value
        if attribute=="sets":return f"region:{key}"
        prefix={"wet_surfaces":"wet_surface","connections":"connection","geometry":"feature","materials":"material","plate_sections":"plate_section","beam_sections":"beam_section","load_cases":"case","supports":"support","masses":"mass","imperfections":"imperfection","coordinate_systems":"coordinate","regions":"region","mesh_records":"mesh","analyses":"analysis","jobs":"job","output_requests":"output_request"}.get(attribute,attribute)
        return f"{prefix}:{key}"

    def _context_menu(self,position):
        index=self.indexAt(position)
        if not index.isValid():return
        if not self.selectionModel().isSelected(index):self.setCurrentIndex(index)
        values=[item.data(Qt.UserRole) for item in self.selectionModel().selectedRows()]
        keys=tuple(key for value in values if (key:=self.item_key(value)))
        if not keys:return
        refs=self.app.geometry_actions.references()
        menu=self.app.geometry_actions.menu_for(refs,self) if refs else QMenu(self)
        menu.addAction(self.app._actions["Hide selected"])
        menu.addAction(self.app._actions["Show selected"])
        menu.addSeparator()
        menu.addAction("Edit / inspect",lambda:self.app.guarded(lambda:self.edit_record(index.siblingAtColumn(0).data(Qt.UserRole)))())
        if not any(key.startswith("members:") for key in keys):
            menu.addAction("Delete",lambda:self.app.guarded(lambda:self.app._delete_tree_items(keys))())
        value=index.siblingAtColumn(0).data(Qt.UserRole)
        if isinstance(value,(tuple,list)) and value[0]=="sets":
            menu.addAction("Rename set…",lambda:self.app.guarded(lambda:self.app.geometry_actions.rename_set(value[1]))())
        if all(key.startswith("feature:") for key in keys):
            for label,action in [("Expose / collapse topology","explode"),("Suppress / resume","suppress"),("Rename","rename"),("Dependencies","dependencies")]:
                menu.addAction(label,lambda checked=False,action=action:self.app.guarded(lambda:self.app._tree_action(action,keys))())
        menu.exec(self.viewport().mapToGlobal(position))
        menu.deleteLater()

    def toggle_feature_topology(self,identifiers):
        identifiers=set(identifiers)
        for identifier in identifiers:self.project.geometry.features.get(identifier)
        exposing=not identifiers.issubset(self.exploded_feature_ids)
        self.exploded_feature_ids=frozenset(set(self.exploded_feature_ids)|identifiers if exposing else set(self.exploded_feature_ids)-identifiers)
        self.refresh()
        return exposing

    def edit_record(self,value):
        if isinstance(value,(tuple,list)) and value[0]=="members":
            return self.app.geometry_actions.properties(self.beam_references(value[1]))
        if not isinstance(value,(tuple,list)):return
        attribute,key=value
        if attribute=="sets":
            self.app.geometry_actions.properties(set_references(self.project,key));return
        if attribute=="geometry":
            if self.app.panels["Construction"].edit_sketch(int(key)):return
            record=self.project.geometry.features.get(int(key));panel=self.app.panels["Geometry"]
            panel.load_values("EditFeature",feature_id=record.feature_id,name=record.name,parameters=record.parameters,inputs=record.inputs,dependencies=record.dependencies)
            page="Geometry"
        elif attribute in {"materials","plate_sections","beam_sections"}:
            values=getattr(self.project,attribute)
            record=next(record for name,record in values.items() if getattr(record,"id",self.project.material_ids.get(name,name))==key)
            operation={"materials":"AddMaterial","plate_sections":"AddPlateSection","beam_sections":"AddBeamSection"}[attribute]
            self.app.panels["Sections"].load_values(operation,**{"material" if attribute=="materials" else "section":record});page="Sections"
        elif attribute in {"supports","masses","wet_surfaces","connections","imperfections","load"}:
            from ...commands import _attribute_container
            record=_attribute_container(self.project,str(key))[3]
            self.app.panels["Loads & BC"].load_values("EditAttribute",replacement=record);page="Loads & BC"
        elif attribute=="output_requests":
            self.app.panels["Definitions"].load_values("EditOutputRequest",request_id=key,replacement=self.project.output_requests[key]);page="Definitions"
        elif attribute=="analyses":
            record=self.project.analyses[key];panel=self.app.panels["Solve"]
            import json
            display={"linear_static":"Linear static","batch_linear_static":"Batch linear static","modal":"Modal","buckling":"Buckling","nonlinear_static":"Nonlinear static","arc_length":"Arc length","transient":"Transient","impact":"Impact","capacity":"Capacity"}.get(record.type,record.type)
            if panel.analysis.findText(display)<0:raise ValueError(f"Analysis type {record.type} is unavailable")
            panel.analysis.setCurrentText(display);panel.options.setPlainText(json.dumps(record.settings));page="Solve"
        else:
            self.app.set_status(f"{attribute.replace('_',' ')}: {key}");return
        self.app.notebook.select(page)

    def sync_from_selection(self):
        if getattr(self,"_tree_driven",False):return
        from PySide6.QtCore import QItemSelection,QItemSelectionModel
        prior=self._refreshing;self._refreshing=True
        try:
            selected=self.selectionModel().selectedRows()
            wanted={index for index in selected if isinstance(index.data(Qt.UserRole),(tuple,list)) and index.data(Qt.UserRole)[0]!="members"}
            refs=set(self.app.selection.items)
            beams=set()
            for ref in refs:
                wanted.update(item.index() for item in self._ref_rows.get(ref,()))
                beams.update(self._ref_beams.get(ref,()))
            for identifier in beams:
                if self._beam_refs[identifier] <= refs:
                    wanted.add(self._rows["Beams"][identifier][0].index())
            current=set(selected)
            for indices,flag in ((current-wanted,QItemSelectionModel.Deselect),(wanted-current,QItemSelectionModel.Select)):
                if not indices:continue
                changes=QItemSelection()
                for index in sorted(indices,key=lambda index:self._row_order.get(index,-1)):changes.select(index,index)
                self.selectionModel().select(changes,flag|QItemSelectionModel.Rows)
        finally:
            self._refreshing=prior

    def refresh_job_states(self):
        """Synchronize job/result rows without traversing geometry."""
        self.project=self.app.project
        jobs=[];results=[]
        for job in self.project.jobs.values():
            stale=self.app._job_is_stale(job)
            state=str(getattr(job.status,"value",job.status))+(" (stale)" if stale else "")
            jobs.append((job.id,job.name,state,("jobs",job.id)))
            if job.result_artifact_id:
                results.append((job.result_artifact_id,job.name,"stale" if stale else "",("result",job.result_artifact_id)))
        self._sync_state_groups((("Jobs",jobs),("Results",results)))

    def refresh_mesh_states(self):
        """Synchronize the small mesh branch without rebuilding topology."""
        self.project=self.app.project
        rows=[]
        for mesh in self.project.mesh_records.values():
            state=getattr(mesh,"status","")
            state=str(getattr(state,"value",state))
            if mesh.source_model_hash!=self.app.session.revision.model_hash:state += " (stale)"
            rows.append((mesh.id,mesh.name,state,("mesh_records",mesh.id)))
        self._sync_state_groups((("Meshes",rows),))

    def _sync_state_groups(self,groups):
        prior=self._refreshing;self._refreshing=True
        try:
            for label,rows in groups:self._sync_group(label,rows)
        finally:self._refreshing=prior
