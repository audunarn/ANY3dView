# ANY3dView 0.5.5

ANY3dView 0.5.5 is the first release distributed under the Mozilla Public
License 2.0. Earlier published versions keep the licence terms supplied with
those versions.

The optional GPU backend no longer depends on the GPL-licensed `tkinter-gl`
Python package. It uses an original, compact Python host for the separately
licensed native TkGL component, whose notice and provenance are included in
the distributions. The public GPU viewer API is unchanged.

This maintenance release also includes the pending legend font-size controls
and delayed scroll highlighting. GPU host lifecycle and destruction are
qualified on Windows before publication.
