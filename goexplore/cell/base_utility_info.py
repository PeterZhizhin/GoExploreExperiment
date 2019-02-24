class BaseUtilityInfo(object):
    """Information that represents how promising a cell is."""

    def utility(self) -> float:
        """Return an info about how promising a cell is.

        Can be used to select most promising cells from an archive.

        :return: A measure of the cell being promising.
        """
        return 0.0
