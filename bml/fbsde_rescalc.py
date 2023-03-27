from .calc import ResCalcMixin
from .equations import FBSDE_LongSin


FBSDE_LongSin_ResCalc = type("FBSDE_LongSin_ResCalc",
                             (FBSDE_LongSin, ResCalcMixin), {})
