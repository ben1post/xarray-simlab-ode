import xsimlab as xs

from xso.core import XSOCore


@xs.process
class Backend:
    """this object contains the backend model and is modified or read by all other components"""

    solver_type = xs.variable(intent='in')
    core = xs.any_object(description='model backend instance is stored here')
    m = xs.any_object(description='math wrapper functions provided by solver')

    def initialize(self):
        """initializing model backend"""
        self.core = XSOCore(self.solver_type)
        self.m = self.core.solver.MathFunctionWrappers

    def finalize(self):
        """finalizing: cleanup"""
        self.core.cleanup()  # for now only affects gekko solve


@xs.process
class Context:
    """ Inherited by all other model components to access backend"""
    core = xs.foreign(Backend, 'core')
    m = xs.foreign(Backend, 'm')

    label = xs.variable(intent='out', groups='label')

    def initialize(self):
        self.label = self.__xsimlab_name__  # assign given label to all subclasses


@xs.process
class FirstInit(Context):
    """ Inherited by all other model components to access backend"""
    group = xs.variable(intent='out', groups='FirstInit')

    def initialize(self):
        super(FirstInit, self).initialize()
        self.group = 1


@xs.process
class SecondInit(Context):
    """ Inherited by all other model components to access backend"""
    firstinit = xs.group('FirstInit')
    group = xs.variable(intent='out', groups='SecondInit')

    def initialize(self):
        super(SecondInit, self).initialize()
        self.group = 2


@xs.process
class ThirdInit(Context):
    """ Inherited by all other model components to access backend"""
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    group = xs.variable(intent='out', groups='ThirdInit')

    def initialize(self):
        super(ThirdInit, self).initialize()
        self.group = 3

@xs.process
class FourthInit(Context):
    """ Inherited by all other model components to access backend"""
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')
    group = xs.variable(intent='out', groups='FourthInit')

    def initialize(self):
        super(FourthInit, self).initialize()
        self.group = 4

@xs.process
class FifthInit(Context):
    """ Inherited by all other model components to access backend"""
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
    """ Solver process executed last """
    firstinit = xs.group('FirstInit')
    secondinit = xs.group('SecondInit')
    thirdinit = xs.group('ThirdInit')
    fourthinit = xs.group('FourthInit')
    fifthinit = xs.group('FifthInit')

    def initialize(self):
        """"""
        # TODO: diagnostic print here
        #print("assembling model")
        #print("SOLVER :", self.core.solver)
        self.core.assemble()

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.core.solve(dt)


@xs.process
class Time(FirstInit):
    """Time is represented as a state variable"""

    time = xs.variable(intent='in', dims='input_time',
                       description='sequence of time points for which to solve the model')
    value = xs.variable(intent='out', dims='time')

    def initialize(self):
        """Initializing Model Time"""
        self.label = self.__xsimlab_name__
        self.core.model.time = self.time

        self.value = self.core.add_variable('time')

        self.core.register_flux(self.label + '_' + self.time_flux.__name__, self.time_flux)
        self.core.add_flux(self.label, 'time', 'time_flux')

    def time_flux(self, **kwargs):
        """Simple linear flux, that represents time within model.
        Necessary for external solvers like odeint.
        """
        dtdt = 1
        return dtdt