import inference.hardmax
import mdp.euclid
import mdp.hardmax

# import inference.hardmax as inf_hardmax  # FAILS!
# TODO: Figure this crazy import mess another day.
# I'm guessing this is because hardmax.occupancy
# which is part of inference.hardmax, is importing
# this module.
import sys
inf_hardmax = sys.modules['pp.inference.hardmax']

inf_default = inf_hardmax  # default type of inference
val_default = mdp.hardmax
