from goexplore.archive import cell
from goexplore.returning import base_policy


class ReplayPolicy(base_policy.BasePolicy):
    """Policy that replays a trajectory to return to a target cell."""
    def return_to_cell(self, expected_cell: cell.Cell):
        """Return to a cell, replaying it's trajectory.

        The environment will be reset, and then the trajectory in the cell
        will be replayed.
        The environment should be deterministic, otherwise it might not work.

        :param expected_cell: Cell that should be returned to
        :return: Latest observation from replaying a trajectory
        """
        state = self._environment.reset()
        for action in expected_cell.trajectory_to_cell:
            state, _, _, _ = self._environment.step(action)
        return state
