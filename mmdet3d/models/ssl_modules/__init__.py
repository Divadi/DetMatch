"""Overall goal of how SSL Modules are organized.
These modules can be organized into two main groups - processors & consumers.
This has nothing to do with semaphores, but I couldn't think of a
good name.
Processors include:
    Taking input data & generating predictions on them.
        Can also store intermediates for faster repeated loss calculation.
    Processing existing elements of batch_dict and applying nms, etc on them.
Consumers include:
    Take existing elements of batch_dict & compute loss on them.
        Can be supervised loss or ssl consistency.
    Metrics/Debugging.
Separation/relationship between the two:
    Processing should be kept away from consumers as much as possible.
        NMS should be done by aprocessor.
    Unless generally applicable (such as saving intermediates or undoing
        augmentations), processors in initialization should take as argument
        a "key" to which to save the results of the processing. Consumers
        should also take as arugment a key to use.
"""

from .consumers import *  # noqa: F401,F403
from .processors import *  # noqa: F401,F403