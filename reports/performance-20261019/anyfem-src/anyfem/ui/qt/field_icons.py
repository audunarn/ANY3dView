"""Semantic SVG artwork for meshing strategies and engineering result fields."""

MESH_DRAWINGS = {
    "quad_first": '<path d="M3 5H23V27H3Z M3 12H23 M3 20H23 M10 5V27 M17 5V27 M23 5 30 16 23 27 M23 12 30 16 23 20"/>',
    "auto": '<path d="M3 7H19V27H3Z M3 17H19 M11 7V27 M19 17 28 27H19 M19 7 28 17 19 17"/><path d="M26 2V10 M22 6H30 M23 3 29 9 M29 3 23 9" stroke="#16804a"/>',
    "mapped": '<path d="M4 6 24 3 29 26 7 29Z M5 14 26 11 M6 22 28 19 M11 5 14 28 M18 4 22 27"/><path d="M2 2H7 M2 2V7 M25 30H30V25" stroke="#16804a"/>',
    "native": '<path d="M3 6 17 3 29 10 26 28 8 29Z M3 6 12 13 17 3 M12 13 29 10 19 20Z M3 6 8 29 12 13 M8 29 19 20 26 28 M19 20 29 10 26 28"/><circle cx="12" cy="13" r="2" fill="#16804a"/>',
}

_AXIS = {"x": "#b44444", "y": "#16804a", "z": "#3565bc"}
_PLATE = '<path d="M3 13 17 7 29 14 15 21Z" fill="#e6eef9"/>'
_BEAM = '<path d="M3 13H29V20H3Z" fill="#e6eef9"/>'


_GLYPHS = {
    "x": "M0 0 4 6 M4 0 0 6", "y": "M0 0 2 3 4 0 M2 3V6",
    "z": "M0 0H4L0 6H4", "u": "M0 0V4Q0 6 2 6Q4 6 4 4V0",
    "r": "M0 6V0H3Q5 0 3 3H0 M2 3 4 6", "|": "M2 0V6",
    "V": "M0 0 2 6 4 0", "M": "M0 6V0L2 3 4 0V6",
    "N": "M0 6V0L4 6V0", "T": "M0 0H4 M2 0V6",
    "P": "M0 6V0H3Q5 0 4 2Q4 3 0 3", "E": "M4 0H0V6H4 M0 3H3",
    "Q": "M0 1Q2-1 4 1V5Q2 7 0 5Z M2 4 5 7",
}


def _label(value):
    # Stroke glyphs remain legible when a host has no matching SVG text font.
    start = (32-(len(value)*6-2))/2
    return ''.join(f'<path d="{_GLYPHS[char]}" transform="translate({start+index*6},25)" stroke-width="1.15"/>'
                   for index,char in enumerate(value))


def field_drawing(name):
    """Return a functional pictogram, or None for other navigation controls."""
    name = str(name).lower()
    if name in MESH_DRAWINGS:return MESH_DRAWINGS[name]
    if name == "magnitude":
        return '<path d="M6 23V5 M6 23H28 M6 23 19 10 M15 10H19V14 M24 19 28 23 24 27 M2 9 6 5 10 9"/>'+_label('|u|')
    if name in ("ux", "uy", "uz"):
        axis = name[-1]
        arrows = {"x": 'M4 16H28 M23 11 28 16 23 21',
                  "y": 'M6 23 25 5 M18 5H25V12',
                  "z": 'M16 24V3 M11 8 16 3 21 8'}
        return f'<path d="{arrows[axis]}" stroke="{_AXIS[axis]}"/><circle cx="16" cy="16" r="3" fill="white"/>'+_label('u'+axis)
    if name in ("rx", "ry", "rz"):
        axis=name[-1]
        return f'<path d="M8 18A10 8 0 1 1 24 18 M24 12V18H30" stroke="{_AXIS[axis]}"/><path d="M16 4V23" stroke-dasharray="2 3"/>'+_label('r'+axis)
    if name == "von_mises":
        return '<path d="M3 12 16 5 29 12 16 22Z" fill="#ffdf85"/><path d="M9 12 16 8 23 12 16 18Z" fill="#e77c48"/><circle cx="16" cy="12" r="2" fill="#b44444"/>'+_label('VM')
    for family in ("membrane", "bending", "shear", "top", "bottom"):
        if name.startswith(family+'_') and name.split('_',1)[1] in ("xx","yy","xy","xz","yz"):
            component=name.split('_',1)[1]
            if family == "membrane":
                body=_PLATE+'<path d="M1 6H10 M6 3 10 6 6 9 M31 22H22 M26 19 22 22 26 25"/>'
            elif family == "bending":
                body='<path d="M3 18Q16 1 29 18 M3 22Q16 5 29 22 M4 9Q8 2 13 7 M10 3 13 7 9 9"/>'
            elif family == "shear":
                body=_PLATE+'<path d="M3 4H20 M16 1 20 4 16 7 M29 24H12 M16 21 12 24 16 27"/>'
            else:
                body='<path d="M3 10 17 4 29 11 15 18Z M3 10V17L15 25 29 18V11 M15 18V25"/>'
                body+=('<path d="M3 10 17 4 29 11 15 18Z" fill="#ecaa52"/>' if family == "top" else
                       '<path d="M3 17 15 25 29 18" stroke="#d47b27" stroke-width="4"/>')
            return body+_label(component)
    if name == "axial_stress":
        return _BEAM+'<path d="M1 7H11 M7 3 11 7 7 11 M31 25H21 M25 21 21 25 25 29"/>'+_label('N')
    if name == "torsional_stress":
        return _BEAM+'<path d="M23 7A6 7 0 1 1 23 24 M23 20V24H28"/>'+_label('T')
    for family in ("bending_stress", "shear_stress"):
        if name in (family+'_y', family+'_z'):
            axis=name[-1]
            body=_BEAM
            body+=('<path d="M7 10Q16-2 25 10 M20 7 25 10 26 5"/>' if family=='bending_stress' else
                   '<path d="M7 3V12 M3 8 7 12 11 8 M25 28V19 M21 23 25 19 29 23"/>')
            return body+_label(('M' if family=='bending_stress' else 'V')+axis)
    if name == "equivalent_plastic_strain":
        return '<path d="M3 6H23V22H3Z" stroke-dasharray="2 2"/><path d="M7 10 29 6 25 25 3 29Z" fill="#e6eef9"/>'+_label('PEEQ')
    return None
