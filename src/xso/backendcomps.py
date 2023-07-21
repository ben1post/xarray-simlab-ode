import xsimlab as xs
from .core import XSOCore


@xs.process
class Backend:
    """Xarray-simlab process initializing and storing model backend and solver.

    This is the interface between the solver and model backend of XSO
    and the model processes, that are decorated as XSO components. As for all
    Xarray-simlab processes, we can define an initialize and finalize method
    to execute at different stages of model runtime.

    Attributes
    __________
    solver_type : xarray-simlab variable
        a string argument passed at model setup, that defines which solver is used
    core : xarray-simlab any_object
        stores the XSOCore class initialized with passed solver_type
    m : xarray-simlab any_object
        store for math function wrappers supplied with Solver contained in XSOCore

    Methods
    _______
    initialize()
        Creates model backend objects at the start of model runtime.
    finalize()
        Calls the cleanup function implemented in XSOCore, relevant to some solvers.
    """

    solver_type = xs.variable(intent='in', description='solver type to use for model')
    core = xs.any_object(description='model backend instance is stored here')
    m = xs.any_object(description='math wrapper functions provided by solver')

    def initialize(self):
        """Initializes model backend at the start of model runtime.

        Creates core attribute to hold XSOCore, and m attribute to hold
        math function wrappers."""
        self.core = XSOCore(self.solver_type)
        self.m = self.core.solver.MathFunctionWrappers

    def finalize(self):
        """Runs finally after model solve to call cleanup function in core"""
        self.core.cleanup()  # currently not implemented in built-in solvers


@xs.process
class Context:
    """Inherited by all other model components to access backend.

    Attributes
    __________
    core : xarray-simlab foreign variable
        Link to core object defined in Backend class
    m : xarray-simlab foreign variable
        Link to math functions wrapper in Backend class
    label : xarray-simlab variable
        Stores label supplied at model setup

    Methods
    -------
    initialize()
        Assigns label given to component, to be referenced in model backend
    """
    core = xs.foreign(Backend, 'core')
    m = xs.foreign(Backend, 'm')

    label = xs.variable(intent='out', groups='label')

    def initialize(self):
        """Every XSO component is initialized with a label attribute
        storing the name supplied at model setup.
        """
        self.label = self.__xsimlab_name__


@xs.process
class FirstInit(Context):
    """Inherits model backend from context and defines initializes stage,
    given to init_stage argument in xso.component decorator.

    This is a hack using xsimlab's group variables to
    force component initialization order.
    """
    group = xs.variable(intent='out', groups='FirstInit')

    def initialize(self):
        super(FirstInit, self).initialize()
        self.group = 1


@xs.process
class SecondInit(Context):
    """Inherits model backend from context and defines initializes stage,
    given to init_stage argument in xso.component decorator.
    """
    firstinit = xs.group('FirstInit')
    group = xs.variable(intent='out', groups='SecondInit')

    def initialize(self):
        super(SecondInit, self).initialize()
        self.group = 2


@xs.process
class ThirdInit(Context):
    """Inherits model backend from context and defines initializes stage,
    given to init_stage argument in xso.component decorator.
    """
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    group = xs.variable(intent='out', groups='ThirdInit')

    def initialize(self):
        super(ThirdInit, self).initialize()
        self.group = 3


@xs.process
class FourthInit(Context):
    """Inherits model backend from context and defines initializes stage,
    given to init_stage argument in xso.component decorator.
    """
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')
    group = xs.variable(intent='out', groups='FourthInit')

    def initialize(self):
        super(FourthInit, self).initialize()
        self.group = 4


@xs.process
class FifthInit(Context):
    """Inherits model backend from context and defines initializes stage,
    given to init_stage argument in xso.component decorator.
    """
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')
    fourthinit = xs.group('FourthInit')
    group = xs.variable(intent='out', groups='FifthInit')

    def initialize(self):
        super(FifthInit, self).initialize()
        self.group = 5


@xs.process
class RunSolver(Context):
    """Inherits model backend from context and calls solver to run
    as final initialization stage of model runtime.
    """
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')
    fourthinit = xs.group('FourthInit')
    fifthinit = xs.group('FifthInit')

    def initialize(self):
        """After all other xso.components were initialized,
        the model can be assembled in core."""
        self.core.assemble()

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.core.solve(dt)


@xs.process
class Time(FirstInit):
    """Process defining model time and registering it in model backend.

    Xsimlab does not provide a time variable by default,
    but since XSO focuses on differential equation modeling,
    this process is included by default.
    """

    time_input = xs.variable(intent='in', dims='time',
                             description='sequence of time for which to solve the model')

    time = xs.index(dims='time')

    def initialize(self):
        """Initializing Time process as fully functional XSO component."""
        super(Time, self).initialize()
        self.label = self.__xsimlab_name__
        self.core.model.time = self.time_input

        self.time = self.core.add_variable('time')

        self.core.register_flux(self.label + '_' + self.time_flux.__name__, self.time_flux)
        self.core.add_flux(self.label, 'time', 'time_flux')

    def time_flux(self, **kwargs):
        """Simple linear flux, that represents time within model.
        Necessary for external solvers like odeint.
        """
        dtdt = 1.
        return dtdt


def create_time_component(time_unit):
    """Helper function to create a Time component with a custom unit registered through the backend."""

    @xs.process
    class Time(FirstInit):
        """Process defining model time and registering it in model backend.
        """

        time_input = xs.variable(intent='in', dims='time',
                                 description='sequence of time for which to solve the model')

        time = xs.index(dims='time', attrs={'units': time_unit})

        def initialize(self):
            """Initializing Time process as fully functional XSO component."""
            # super(Time, self).initialize()
            self.label = self.__xsimlab_name__

            self.core.model.time = self.time_input

            self.time = self.core.add_variable('time')

            self.core.register_flux(self.label + '_' + self.time_flux.__name__, self.time_flux)
            self.core.add_flux(self.label, 'time', 'time_flux')

        def time_flux(self, **kwargs):
            """Simple linear flux, that represents time within model.
            Necessary for external solvers like odeint.
            """
            dtdt = 1.
            return dtdt

    return Time
