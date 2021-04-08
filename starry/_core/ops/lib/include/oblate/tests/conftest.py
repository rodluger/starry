import oblate
import pytest


@pytest.fixture(scope="session", autouse=True)
def solvers(lmax=8):
    cppsolver = oblate.CppSolver(lmax)
    numsolver = oblate.NumericalSolver(lmax)
    pysolver = oblate.PythonSolver(lmax)
    brutesolver = oblate.BruteSolver(lmax)
    return {
        "cpp": cppsolver,
        "num": numsolver,
        "py": pysolver,
        "brute": brutesolver,
    }
