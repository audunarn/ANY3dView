"""Neutral file-inspector results using ANYfileio's authoritative readers."""
from dataclasses import dataclass,field
from pathlib import Path
from anyfileio import describe,read,read_sesam_fem_document,read_sesam_semantics,get_element_spec
from anyfileio.sesam.validation import validate_sesam_fem_document
from anyfileio.sesam.sif import read_sesam_sif_stress
from anyfileio.diagnostics import FemDiagnostic,FileFormatError


@dataclass
class FileInspection:
    path: Path
    kind: str
    summary: dict=field(default_factory=dict)
    diagnostics: list=field(default_factory=list)
    records: list=field(default_factory=list)
    document: object=None

    def report(self):
        return dict(source=str(self.path),summary=self.summary,diagnostics=[item.as_dict() for item in self.diagnostics])


def inspect_file(path):
    target=Path(path);result=FileInspection(target,describe(target))
    if target.suffix.lower() in {".fem",".sif"}:
        document=read_sesam_fem_document(target,strict=False);result.document=document
        result.diagnostics=list(document.diagnostics)+list(validate_sesam_fem_document(document))
        result.summary={key:len(getattr(document,key)) for key in ("nodes","elements","materials","sections","boundaries","load_records","dependencies","unknown_records")}
        result.summary["records"]=len(document.raw_records);histogram={}
        for element in document.elements.values():
            spec=get_element_spec(element.type_code);label=f"{element.type_code} ({spec.name if spec else 'unsupported'})"
            histogram[label]=histogram.get(label,0)+1
        result.summary["element_types"]=histogram
        for record in document.raw_records[:5000]:
            values=", ".join(f"{value:g}" for value in record.numeric_fields[:6])
            result.records.append((record.name,f"{record.source_line_start}-{record.source_line_end}",values+" "+" ".join(record.text_fields[:2])))
        try:
            semantics=read_sesam_semantics(document,strict=False)
            result.summary["mesh"]={key:len(getattr(semantics.mesh,key)) for key in ("quads","tris","beams")}
            result.summary["mesh"]["supports"]=len(semantics.supports)
        except (ValueError,FileFormatError):pass
        if target.suffix.lower()==".sif":
            try:
                stress=read_sesam_sif_stress(target)
                result.summary["results"]=dict(components=list(stress.components),nodal_stress=len(stress.nodal_stress),element_stress=len(stress.element_stress),units=stress.units)
            except (OSError,ValueError,FileFormatError) as error:result.diagnostics.append(FemDiagnostic("SIF900",str(error),severity="warning"))
    else:
        parsed=read(target)
        if isinstance(parsed,dict):result.summary=parsed
        else:
            result.summary=parsed.summary()
            result.summary["rotations"]="Absent in CalculiX result files"
            result.diagnostics=[FemDiagnostic("CCX900",message,severity="warning") for message in parsed.warnings]
    return result
