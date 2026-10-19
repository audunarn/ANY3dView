import sys
sys.path.insert(0, r"C:\Github\ANY3dView\reports\performance-20261019\worker")
from patch_helper import patch

PATH = r"C:\Github\ANY3dView\src\any3dview\qt.py"

patch(PATH, [
    # _SoftwareResources: mask revision counter for retained preparation.
    (
        "    def __init__(self):\r\n"
        "        self.appearance = {}\r\n"
        "        self.masks = {}\r\n"
        "        self.chunk_masks = {}\r\n"
        "        self.chunk_pickable = {}\r\n",
        "    def __init__(self):\r\n"
        "        self.appearance = {}\r\n"
        "        self.masks = {}\r\n"
        "        self.chunk_masks = {}\r\n"
        "        self.chunk_pickable = {}\r\n"
        "        # Bumped on every mask/appearance mutation so retained software\r\n"
        "        # preparation can invalidate without introspecting contents.\r\n"
        "        self.mask_revision = 0\r\n",
    ),
    (
        "    def add_mesh(self, handle, **appearance):\r\n"
        "        self.appearance[id(handle)] = appearance\r\n"
        "        self.geometry_uploads += 1\r\n",
        "    def add_mesh(self, handle, **appearance):\r\n"
        "        self.appearance[id(handle)] = appearance\r\n"
        "        self.geometry_uploads += 1\r\n"
        "        self.mask_revision += 1\r\n",
    ),
    (
        "            if key[0] == id(handle):\r\n"
        "                self.chunk_masks.pop(key, None)\r\n"
        "                self.chunk_pickable.pop(key, None)\r\n",
        "            if key[0] == id(handle):\r\n"
        "                self.chunk_masks.pop(key, None)\r\n"
        "                self.chunk_pickable.pop(key, None)\r\n"
        "        self.mask_revision += 1\r\n",
    ),
    (
        "    def set_semantic_masks(self, handle, **masks):\r\n"
        "        self.masks[id(handle)] = masks\r\n",
        "    def set_semantic_masks(self, handle, **masks):\r\n"
        "        self.masks[id(handle)] = masks\r\n"
        "        self.mask_revision += 1\r\n",
    ),
    (
        "    def set_highlighted_elements(self, handle, indices):\r\n"
        "        self.masks.setdefault(id(handle), {})[\"highlighted_elements\"] = indices\r\n",
        "    def set_highlighted_elements(self, handle, indices):\r\n"
        "        self.masks.setdefault(id(handle), {})[\"highlighted_elements\"] = indices\r\n"
        "        self.mask_revision += 1\r\n",
    ),
    (
        "    def set_chunk_semantic_masks(self, handle, chunk_id, **masks):\r\n"
        "        self.chunk_masks[id(handle), chunk_id] = masks\r\n",
        "    def set_chunk_semantic_masks(self, handle, chunk_id, **masks):\r\n"
        "        self.chunk_masks[id(handle), chunk_id] = masks\r\n"
        "        self.mask_revision += 1\r\n",
    ),
    (
        "    def clear_chunk_semantic_masks(self, handle, chunk_id):\r\n"
        "        self.chunk_masks.pop((id(handle), chunk_id), None)\r\n",
        "    def clear_chunk_semantic_masks(self, handle, chunk_id):\r\n"
        "        self.chunk_masks.pop((id(handle), chunk_id), None)\r\n"
        "        self.mask_revision += 1\r\n",
    ),
    (
        "    def set_chunk_pickable(self, handle, chunk_id, enabled):\r\n"
        "        self.chunk_pickable[id(handle), chunk_id] = enabled\r\n",
        "    def set_chunk_pickable(self, handle, chunk_id, enabled):\r\n"
        "        self.chunk_pickable[id(handle), chunk_id] = enabled\r\n"
        "        self.mask_revision += 1\r\n",
    ),
])
