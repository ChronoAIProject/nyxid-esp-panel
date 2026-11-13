// The authenticated ESP-SR 2.6.1 P4 WakeNet library allocates this exact size.
// ESP-DL 3.3.15 added a member and breaks that precompiled caller's object ABI.
#include "dl_model_base.hpp"
static_assert(sizeof(dl::Model) == 144, "ESP-DL Model ABI differs from the pinned WakeNet caller");
