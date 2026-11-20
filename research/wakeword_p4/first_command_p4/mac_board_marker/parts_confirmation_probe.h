#pragma once

// Called once per panel read, immediately before the pending score pair is reported.
// At most one complete 480-sample detector call fits in a 256-sample panel read.
extern "C" void report_confirmation(void) {
    using namespace ordered_parts;
    if (!pending_tick) return;
    constexpr unsigned span = 5;
    constexpr unsigned cooldown = 67;
    static unsigned previous_hit = 0, broad_hit = 0, last_event = 0;
    static bool active = false;
    const bool previous = pending_scores[0] >= 1638;
    const bool broad = pending_scores[1] >= -1560;
    if (previous) previous_hit = pending_tick;
    if (broad) broad_hit = pending_tick;
    const bool confirmed = (previous && pending_tick - broad_hit <= span) ||
                           (broad && pending_tick - previous_hit <= span);
    if (confirmed && !active && pending_tick - last_event >= cooldown) {
        const unsigned sample = (pending_tick + 1) * 480 - 81600;
        printf("CONFIRM %u\n", sample);
        last_event = pending_tick;
    }
    active = confirmed;
}
