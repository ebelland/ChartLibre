"""The calculations behind the series operations, without any Qt.

Each module takes plain arrays and parameters and returns plain results, so
the same code runs behind a dialog, in a test, in a script or on a worker
thread. A dialog's job is to collect the parameters and present the result;
the arithmetic lives here. See todo.txt R-01.
"""


class Stopped(BaseException):
    """A calculation was asked to stop before it finished.

    Raised by an engine when its ``should_stop`` callback says so - how an
    operation running in the background is stopped. A BaseException, like
    KeyboardInterrupt, on purpose: the optimisers and searches catch
    Exception around every evaluation of a model - a sample the model cannot
    evaluate is "bad, move on" - and would swallow a stop request as one
    more bad sample.
    """
