import oblate
import pytest


@pytest.fixture(scope="session", autouse=True)
def solvers(lmax=5):
    cppsolver = oblate.CppSolver(lmax, linear=True)
    cppexactsolver = oblate.CppSolver(lmax, linear=False)
    numsolver = oblate.NumericalSolver(lmax, linear=False)
    linsolver = oblate.NumericalSolver(lmax, linear=True)
    pysolver = oblate.PythonSolver(lmax)
    brutesolver = oblate.BruteSolver(lmax)
    return {
        "cpp": cppsolver,
        "cppx": cppexactsolver,
        "num": numsolver,
        "lin": linsolver,
        "py": pysolver,
        "brute": brutesolver,
    }
