import dataclasses
import typing

from goexplore.cell import base_returning_info


@dataclasses.dataclass
class LoadableCellInfo(base_returning_info.BaseReturningInfo):
    snapshot_data: typing.Any = None
