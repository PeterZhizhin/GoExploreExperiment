import typing
import dataclasses

@dataclasses.dataclass
class Cell(object):
    # An iterable with actions that lead to the cell
    trajectory_to_cell: typing.Iterable[typing.Any] = dataclasses.field(
        default_factory=list)
