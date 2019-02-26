import dataclasses

from goexplore.cell import base_prioritizing_info


@dataclasses.dataclass()
class RewardPrioritization(base_prioritizing_info.BasePrioritizingInfo):
    """Prioritization based on the reward of the cell."""
    reward: float
    trajectory_length: int

    def __lt__(self, other: RewardPrioritization):
        if self.reward == other.reward:
            return self.trajectory_length > other.trajectory_length
        return self.reward < other.reward
