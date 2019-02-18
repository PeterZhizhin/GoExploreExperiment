import typing
import dataclasses

@dataclasses.dataclass
class Cell(object):
    trajectory_to_cell: typing.Iterable[typing.Any] = dataclasses.field(
        default_factory=list)
