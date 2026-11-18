#include "dl_fft_base.h"

// The P4 QEMU release traps on esp.lp.setup in these two kernels.
void __wrap_dl_fft2r_fc32_arp4_(float *data, int n, float *table)
{
    dl_fft2r_fc32_ansi(data, n, table);
}

void __wrap_dl_ifft2r_fc32_arp4_(float *data, int n, float *table)
{
    dl_ifft2r_fc32_ansi(data, n, table);
}
