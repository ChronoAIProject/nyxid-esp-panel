#pragma once

// Bound the per-request JPEG response; there is no persistent image cache.
#define SNAPSHOT_MAX_BYTES (512u * 1024u)
