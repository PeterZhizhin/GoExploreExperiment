import typing
import dataclasses

import gym


@dataclasses.dataclass
class Cell(object):
    full_state: typing.Optional[gym.Space] = None
    reward: float = 0
    steps_to_cell: int = 0
    trajectory_to_cell: typing.Iterable[typing.Any] = dataclasses.field(
        default_factory=list)

    cell_representation: typing.Hashable = None

    number_of_visits: int = 0
    total_exploration_episodes: int = 0
    visits_since_last_success: int = 0
