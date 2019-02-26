import typing
import gym

from goexplore.cell import total_cell

TrajectoryType = typing.Iterable[typing.Tuple[gym.Space,  # State
                                              gym.Space,  # Action
                                              float,  # Reward
                                              gym.Space,  # New state
                                              bool,  # Is done
                                              typing.Optional[object],  # Info
                                              ]]


class BaseCellFactory(object):
    """Factory that can create cells based on cells and trajectories."""

    def generate_cells(self, cell: total_cell.Cell, trajectory: TrajectoryType,
                       env: gym.Env) -> typing.Iterable[total_cell.Cell]:
        """Generate new cells from a cell and a trajectory.

        This method is called as the environment is being explored.

        :param cell: Cell that was explored from.
        :param trajectory: Trajectory of actions that leads to the new cell.
        :param env: Environment that is being explored, the trajectory leads
            to this state of the environment.
        :return: Iterable of newly created cells.
        """
        raise NotImplementedError
