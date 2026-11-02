#pragma once
#include <stdbool.h>
#include "cJSON.h"
// Called only by the Nyx consumer task. The replay window survives reconnects.
bool nyx_security_verify_proxy(cJSON *root, const char *secret);
bool nyx_security_verify_upload(cJSON *root, const char *secret, const char *node_id);
