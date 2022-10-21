import pytest
import xso
import xsimlab as xs
from xso.backendcomps import Backend

# TODO:
#   - add tests for each variable type
#   - add tests for XSOCore class, Model class
#   - add tests for Solvers
#   - add tests for different usages of components within a model
#   - add tests for different Model inputs

# The way I wrote the framework seems to be not very amendable to testing, or
# am I seriously misunderstanding something
# One Tip: Write failing test, then modify code to pass test
#
# Unit testing is for testing the things the code is supposed to do
#   testing the different inputs it can receive
#   testing the types of usages it should be able to render
#
#

# TODO:
#   - add tests for supplying similarly named variables both at component construction
#   - and at model setup, this needs to raise an understandable error in code, and check that here


@pytest.fixture
def backend():
    # create and initialize backend to be able to initialize test component
    example_backend = Backend(solver_type='stepwise')
    example_backend.initialize()
    return example_backend


def test_component(backend):
    # test constructor

    @xso.component
    class TestComp:
        # basic class using two types of variables
        some_var = xso.variable()
        another_var = xso.variable(foreign=True)

    test_comp = TestComp(m=backend.core, some_var_label='X', some_var_init=2, another_var='Y')
    test_comp.__xsimlab_name__ = 'TEST'

    test_comp.initialize()
    print("hello")
    assert test_comp.some_var_init == 2
