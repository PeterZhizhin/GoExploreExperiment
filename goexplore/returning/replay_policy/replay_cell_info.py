import dataclasses
import typing

from goexplore.cell import base_returning_info


@dataclasses.dataclass
class ReplayCellInfo(base_returning_info.BaseReturningInfo):
    trajectory: typing.Collection[typing.Any] = dataclasses.field(
        default_factory=list)
