from any3dview import (
    PickBinding,
    ProjectedPrimitive,
    ProjectedSelectionIndex,
    SelectionDepth,
    SelectionFilter,
)


def square(index, key, depth):
    return ProjectedPrimitive(
        index,
        "polygon",
        ((10, 10), (90, 10), (90, 90), (10, 90)),
        (depth,) * 4,
        PickBinding.one(key, "mesh.element"),
    )


def test_projected_selection_orders_visible_then_occluded_hits():
    index = ProjectedSelectionIndex(
        [square(0, "front", 2), square(1, "back", 5)], 100, 100
    )
    hits = index.point_hits(50, 50, SelectionFilter(), radius=0)
    assert [(hit.key, hit.visible) for hit in hits] == [
        ("front", True),
        ("back", False),
    ]


def test_visible_and_through_region_policies_differ():
    index = ProjectedSelectionIndex(
        [square(0, "front", 2), square(1, "back", 5)], 100, 100
    )
    visible = index.rectangle_hits(
        (0, 0, 100, 100), SelectionFilter(), crossing=False, depth=SelectionDepth.VISIBLE
    )
    through = index.rectangle_hits(
        (0, 0, 100, 100), SelectionFilter(), crossing=False, depth=SelectionDepth.THROUGH
    )
    assert [hit.key for hit in visible] == ["front"]
    assert {hit.key for hit in through} == {"front", "back"}
