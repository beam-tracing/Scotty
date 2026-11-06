# -*- coding: utf-8 -*-
# Copyright 2021 - 2026, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0
"""Moved to `scotty.antenna.horns`. This name still works: it is the same
module, so ``from scotty.hornpy import ...`` gives the same objects"""

import sys

from scotty.antenna import horns

sys.modules[__name__] = horns
