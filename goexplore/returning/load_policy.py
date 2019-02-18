from goexplore.archive import loadable_cell
from goexplore.envs import save_loadable
from goexplore.returning import base_policy


class LoadPolicy(base_policy.BasePolicy):
    """A class that returns to expected state using loading."""

    def __init__(self, environment: save_loadable.SaveLoadableEnv):
        """Create a load policy using save-loadable environment.

        :param environment: Save-loadable environment
        :raises TypeError: If an environment is not SaveLoadable.
        """
        super().__init__(environment)
        if not isinstance(environment, save_loadable.SaveLoadableEnv):
            raise TypeError(
                "Provided environment should be a subclass "
                "of SaveLoadableEnv in order to use "
                "LoadPolicy for returning, got {!r}".format(environment))

    def return_to_cell(self, expected_cell: loadable_cell.LoadableCell):
        """Returning to cell loading from snapshot.

        :param expected_cell: Cell to return to, should be loadable.
        :return: Observation from the loaded cell
        :raises TypeError: If expected_cell is not a LoadableCell.
        """
        if not isinstance(expected_cell, loadable_cell.LoadableCell):
            raise TypeError("Provided cell should be a subclass "
                            "of LoadableCell, got {!r}".format(expected_cell))
        return self._environment.load_snapshot(expected_cell.snapshot_data)
