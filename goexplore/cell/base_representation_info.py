class BaseRepresentationInfo(object):
    """A class that is used as keys in an archive."""

    def __hash__(self):
        """Used to store elements in a dictionary.

        All elements have same hash by default.
        """
        return 0
