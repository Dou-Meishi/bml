from .calc import ResCalcMixin
from .equations import FBSDE_LongSin, FBSDE_FuSinCos, FBSDE_JiLQ5


FBSDE_LongSin_ResCalc = type("FBSDE_LongSin_ResCalc",
                             (FBSDE_LongSin, ResCalcMixin), {})
FBSDE_FuSinCos_ResCalc = type("FBSDE_FuSinCos_ResCalc",
                             (FBSDE_FuSinCos, ResCalcMixin), {})
FBSDE_JiLQ5_ResCalc = type("FBSDE_JiLQ5_ResCalc",
                           (FBSDE_JiLQ5, ResCalcMixin), {})
