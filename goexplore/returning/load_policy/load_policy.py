from goexplore.envs import save_loadable
from goexplore.returning import base_policy
from goexplore.returning.load_policy import loadable_cell_info


class LoadPolicy(base_policy.BasePolicy):
    """A class that returns to expected state using loading."""

    def __init__(self, environment: save_loadable.SaveLoadableEnv):
        """Create a load policy using a save-loadable environment.

        :param environment: Save-loadable environment
        :raises TypeError: If provided environment is not SaveLoadable.
        """
        super().__init__(environment)
        if not isinstance(environment, save_loadable.SaveLoadableEnv):
            raise TypeError(
                "Provided environment should be a subclass "
                "of SaveLoadableEnv in order to use "
                "LoadPolicy for returning, got {!r}".format(environment))

    def return_to_cell(self,
                       expected_cell: loadable_cell_info.LoadableCellInfo):
        """Returning to a cell loading from a snapshot.

        The cell should contain the snapshot.

        :param expected_cell: Cell to return to, should be loadable.
        :return: Observation from the loaded cell
        :raises TypeError: If expected_cell is not a LoadableCell.
        """
        if not isinstance(expected_cell, loadable_cell_info.LoadableCellInfo):
            raise TypeError(
                "Provided cell return info should be a subclass "
                "of LoadableCellInfo, got {!r}".format(expected_cell))
        return self._environment.load_snapshot(expected_cell.snapshot_data)
