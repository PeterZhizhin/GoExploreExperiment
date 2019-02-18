import typing
import dataclasses

from goexplore.archive import cell


@dataclasses.dataclass
class LoadableCell(cell.Cell):
    # Some information that can be used to reset an environment to the cell
    snapshot_data: typing.Any = None
