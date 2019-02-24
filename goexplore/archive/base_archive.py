import typing

from goexplore.cell import total_cell


class BaseArchive(object):
    """Archive that contains cells, can update and batch them."""

    def update(self, new_cell: total_cell.Cell):
        """Given new cell, update the archive.

        If the cell is not present in the archive, then add it.
        If the cell is present in the archive, then keep the one with higher
        reward and shorter trajectory. Also, update the parameters of the cell
        (e.g. number of visits, number of visits since last updated).

        :param new_cell: Cell that should be used to update the archive.
        """
        raise NotImplementedError

    def sample(self, batch_size: int = 1) -> typing.Iterable[total_cell.Cell]:
        """Sample a batch of cells.

        If the archive is empty, a ValueError is raised.

        :param batch_size: Size of the batch to be sampled.
        :return: A batch of cells, can be smaller than desired batch size.
        :raises ValueError, if the archive is empty
        """
        raise NotImplementedError
