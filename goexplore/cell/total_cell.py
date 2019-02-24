from __future__ import annotations

import dataclasses

from goexplore.cell import base_returning_info
from goexplore.cell import base_prioritizing_info
from goexplore.cell import base_representation_info
from goexplore.cell import base_utility_info


@dataclasses.dataclass
class Cell(object):
    """Cell for Go-Explore algorithm."""

    # This is used in order to return to the cell.
    returning_info: base_returning_info.BaseReturningInfo = dataclasses.field(
        default_factory=base_returning_info.BaseReturningInfo)
    prioritizing_info: base_prioritizing_info.BasePrioritizingInfo = dataclasses.field(
        default_factory=base_prioritizing_info.BasePrioritizingInfo)
    representation_info: base_representation_info.BaseRepresentationInfo = dataclasses.field(
        default_factory=base_representation_info.BaseRepresentationInfo)
    utility_info: base_utility_info.BaseUtilityInfo = dataclasses.field(
        default_factory=base_utility_info.BaseUtilityInfo)
