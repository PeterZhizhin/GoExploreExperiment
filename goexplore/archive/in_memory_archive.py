import typing

import numpy as np

from goexplore.archive import base_archive
from goexplore.cell import total_cell
from goexplore.cell import base_representation_info


class InMemoryArchive(base_archive.BaseArchive):
    """Archive that keeps all cells in RAM."""

    def __init__(self):
        # Mapping from cell representation to an actual cell.
        self._archive = dict(
        )  # type: typing.Dict[base_representation_info.BaseRepresentationInfo, total_cell.Cell]

    def update(self, new_cell: total_cell.Cell):
        new_cell_representation = new_cell.representation_info

        if new_cell_representation not in self._archive:
            self._archive[new_cell_representation] = new_cell
            return

        existing_cell = self._archive[
            new_cell_representation]  # type: total_cell.Cell

        new_cell_priority = new_cell.prioritizing_info
        existing_cell_priority = existing_cell.prioritizing_info
        if existing_cell_priority < new_cell_priority:
            # Keep existing utility information as is is used to
            # sample new cells.
            new_cell.utility_info = existing_cell.utility_info
            self._archive[new_cell_representation] = new_cell
            return

    def sample(self, batch_size: int = 1) -> typing.Iterable[total_cell.Cell]:
        if not self._archive:
            raise ValueError(("Trying to sample a batch of size {} from "
                              "an empty archive.").format(batch_size))
        cells_array = list(self._archive.values())
        cells_utilities = np.array(
            [cell.utility_info.utility() for cell in cells_array])

        # Gumbel-max trick for softmax sampling
        gumbel_shifts = np.random.gumbel(
            size=(batch_size, cells_utilities.size))
        gumbel_shifted_utilities = gumbel_shifts + cells_utilities

        chosen_indexes = gumbel_shifted_utilities.argmax(axis=1)
        return (cells_array[index] for index in chosen_indexes)
