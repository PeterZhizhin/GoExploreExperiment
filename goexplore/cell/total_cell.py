from __future__ import annotations

import typing
import dataclasses

from goexplore.cell import base_returning_info


@dataclasses.dataclass
class Cell(object):
    """Cell for Go-Explore algorithm."""

    # This is used in order to return to the cell.
    returning_info: base_returning_info.BaseReturningInfo
