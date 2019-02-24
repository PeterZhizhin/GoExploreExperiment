class BasePrioritizingInfo(object):
    """Base information for prioritizing cells.

    The class can contain different statistics required to select most
    appropriate cell.
    """

    def __lt__(self, other):
        """Whether the cell should be prioritized for storing it in an archive.

        :param other: Other cell to compare
        :return: bool
        """
        raise NotImplementedError
