__all__ = ["Tolerance"]
from numpy import finfo, float64


class Tolerance:
    @staticmethod
    def atol():
        """
        Static absolute tolerance method
        Returns
        -------
        atol : library-wide set absolute tolerance for kernels
        """
        return 1e-7

    @staticmethod
    def rtol():
        """
        Static relative tolerance method
        Returns
        -------
        tol : library-wide set relative tolerance for kernels
        """
        return finfo(float64).eps * 1e11
