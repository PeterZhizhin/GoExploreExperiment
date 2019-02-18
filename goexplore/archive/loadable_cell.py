import typing
import dataclasses

from goexplore.archive import cell


@dataclasses.dataclass
class LoadableCell(cell.Cell):
    snapshot_data: typing.Any = None
