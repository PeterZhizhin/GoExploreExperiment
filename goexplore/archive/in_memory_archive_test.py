import unittest
import dataclasses
import numpy as np

from goexplore.archive import in_memory_archive
from goexplore.cell import total_cell
from goexplore.cell import base_representation_info
from goexplore.cell import base_prioritizing_info
from goexplore.cell import base_utility_info


class InMemoryArchiveTestCase(unittest.TestCase):
    @dataclasses.dataclass(unsafe_hash=True)
    class TestRepresentationInfo(
            base_representation_info.BaseRepresentationInfo):
        for_hashing: int

    @dataclasses.dataclass()
    class TestPrioritizingInfo(base_prioritizing_info.BasePrioritizingInfo):
        compare_value: float = 0.0

        def __lt__(self, other):
            return self.compare_value < other.compare_value

    @dataclasses.dataclass()
    class TestUtilityInfo(base_utility_info.BaseUtilityInfo):
        ret_utility: float

        def utility(self):
            return self.ret_utility

    def setUp(self):
        self.archive = in_memory_archive.InMemoryArchive()

        self.cells_100_different = [
            total_cell.Cell(
                prioritizing_info=self.TestPrioritizingInfo(compare_value=i),
                representation_info=self.TestRepresentationInfo(for_hashing=i),
                utility_info=self.TestUtilityInfo(ret_utility=i))
            for i in range(100)
        ]

    def test_new_cell_gets_sampled_in_archive(self):
        cell = total_cell.Cell()
        self.archive.update(cell)
        sampled_cells = self.archive.sample(batch_size=1)

        self.assertEqual(next(sampled_cells), cell)
        with self.assertRaises(
                StopIteration,
                msg='Expected that there is only one element in '
                'the returned cells batch.'):
            next(sampled_cells)

    def test_sample_from_empty_archive_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, '.*empty.*'):
            self.archive.sample(batch_size=10)

    def test_sample_with_large_number_of_cells_return_desired_batch_size(self):
        for cell in self.cells_100_different:
            self.archive.update(cell)

        self.assertEqual(len(list(self.archive.sample(10))), 10)
        self.assertEqual(len(list(self.archive.sample(100))), 100)
        self.assertEqual(len(list(self.archive.sample(1000))), 1000)

    def test_sample_cells_that_large_utility_only_large_utility_returned(self):
        # Only these cells should be selected
        for cell in self.cells_100_different[:10]:
            cell.utility_info.ret_utility = 1e100

        for cell in self.cells_100_different[10:]:
            cell.utility_info.ret_utility = 0

        for cell in self.cells_100_different:
            self.archive.update(cell)

        sampled_cells = self.archive.sample(10)
        for cell in sampled_cells:
            self.assertEqual(
                cell.utility_info.ret_utility,
                1e100,
                msg='Only cells with 1e100 utility should be selected')

    def test_sample_cells_returns_values_according_to_utilities_as_softmax(
            self):
        probabilities = [0.1, 0.7, 0.2]
        total_trials = 10**5

        cells = [
            total_cell.Cell(
                representation_info=self.TestRepresentationInfo(i),
                utility_info=self.TestUtilityInfo(
                    # Log for softmax (e^utility is used for scaling)
                    np.log(prob))) for i, prob in enumerate(probabilities)
        ]
        for cell in cells:
            self.archive.update(cell)

        sampled_cells = self.archive.sample(batch_size=total_trials)

        successes = [0] * len(probabilities)
        for cell in sampled_cells:
            successes[cell.representation_info.for_hashing] += 1

        for desired_prob, success_count in zip(probabilities, successes):
            got_prob = float(success_count) / total_trials

            abs_difference = np.abs(got_prob - desired_prob)
            max_difference = 9 * np.sqrt(desired_prob *
                                         (1 - desired_prob) / total_trials)

            # Probability of failing here is ~1e-15
            # (if the implementation is valid).
            self.assertLess(abs_difference, max_difference)

    def test_new_cells_with_same_representation_only_best_returned_on_sample(
            self):
        worse_cell = total_cell.Cell(
            representation_info=self.TestRepresentationInfo(0),
            prioritizing_info=self.TestPrioritizingInfo(compare_value=0))

        better_cell = total_cell.Cell(
            representation_info=self.TestRepresentationInfo(0),
            prioritizing_info=self.TestPrioritizingInfo(compare_value=100))

        self.archive.update(worse_cell)
        self.archive.update(better_cell)

        sample = self.archive.sample(batch_size=1)
        self.assertEqual(next(sample), better_cell)

    def test_same_representation_better_priority_utility_stays_same(self):
        worse_cell = total_cell.Cell(
            representation_info=self.TestRepresentationInfo(0),
            prioritizing_info=self.TestPrioritizingInfo(compare_value=0),
            utility_info=self.TestUtilityInfo(ret_utility=100))

        better_cell = total_cell.Cell(
            representation_info=self.TestRepresentationInfo(0),
            prioritizing_info=self.TestPrioritizingInfo(compare_value=100),
            utility_info=self.TestUtilityInfo(ret_utility=10))

        self.archive.update(worse_cell)
        self.archive.update(better_cell)

        sample = next(self.archive.sample(batch_size=1))
        self.assertEqual(
            sample.utility_info.ret_utility,
            100,
            msg='Expected that when a cell is replaced with '
            'a better one, utility for sampling stays the same.')


if __name__ == "__main__":
    unittest.main()
