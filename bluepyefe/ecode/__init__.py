"""eCode init script"""

"""
Copyright (c) 2020, EPFL/Blue Brain Project
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>
 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.
 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

from . import DeHyperPol
from . import HyperDePol
from . import SpikeRec
from . import negCheops
from . import posCheops
from . import ramp
from . import sAHP
from . import sineSpec
from . import step

# The ecode names have to be lower case only to avoid having to
# define duplicates for upper and lower cases.
eCodes = {
    "spontaneous": step.Step,
    "idrest": step.Step,
    "idthresh": step.Step,
    "idthres": step.Step,
    "apwaveform": step.Step,
    "iv": step.Step,
    "step": step.Step,
    "spontaps": step.Step,
    "firepattern": step.Step,
    "sponnohold30": step.Step,
    "sponhold30": step.Step,
    "spontnohold30": step.Step,
    "sponthold30": step.Step,
    "starthold": step.Step,
    "startnohold": step.Step,
    "delta": step.Step,
    "sahp": sAHP.SAHP,
    "idhyperpol": sAHP.SAHP,
    "irdepol": sAHP.SAHP,
    "irhyperpol": sAHP.SAHP,
    "iddepol": sAHP.SAHP,
    "ramp": ramp.Ramp,
    "ap_thresh": ramp.Ramp,
    "apthresh": ramp.Ramp,
    "hyperdepol": HyperDePol.HyperDePol,
    "dehyperpol": DeHyperPol.DeHyperPol,
    "negcheops": negCheops.NegCheops,
    "poscheops": posCheops.PosCheops,
    "spikerec": SpikeRec.SpikeRec,
    "sinespec": sineSpec.SineSpec,
}
