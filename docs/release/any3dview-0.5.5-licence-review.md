# ANY3dView 0.5.5 licence review

Date: 2026-09-03

- Repository history attribution was reviewed before relicensing. Commits are
  attributed to Audun Arnesen Nyhus, apart from automated GitHub Actions bot
  commits.
- Project metadata, README, full licence text, wheel licence metadata, and
  source-distribution contents declare MPL-2.0.
- The GPL-licensed `tkinter-gl` Python package is no longer a dependency and
  none of its Python modules is bundled.
- The native TkGL 1.2.1 component is distributed under the separate permissive
  terms retained verbatim in `src/any3dview/gpu/tkgl/license.terms`.
- Native component provenance is recorded in
  `src/any3dview/gpu/tkgl/PROVENANCE.md` and the user-facing inventory is in
  `THIRD_PARTY_NOTICES.md`.
- The release licensing check prevents restoration of the removed Python
  dependency or wrapper import.
- Previously published versions retain their historical licence terms.
