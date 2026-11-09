"""Confusing phrases; full Hey Nyx/Nix/Nicks aliases are always positive."""
HARD_NEGATIVES = [
    "Hey Nick.", "Hey Next.", "Hey Mix.", "Hey Six.", "Hey Max.",
    "Hey Nice.", "Hey Nexus.", "Hey Alex.", "Hi ESP.", "Hey Siri.",
    "Hey Nick, turn on the light.", "Hey Next, what time is it?",
    "The next thing is ready.", "I mixed the ingredients.",
    "I heard Nick speaking.", "Take the next exit.", "Hello, can you hear me?",
    "How can I help you?", "Please check the volume.", "Thanks, that is all.",
]

# The target is the complete two-word phrase. Its isolated components are not
# full wake phrases. Keep these explicit partial negatives separate from the
# original recipe; never relabel any full Hey Nyx/Nix/Nicks utterance.
PARTIAL_WAKE_NEGATIVES = [
    "Hey.", "Hey!", "Hey there.", "Nicks.", "Nix.", "Nyx.",
    "Nicks, turn on the light.", "Nix the plan.",
    "The nicks on the surface.", "Ask Nicks about it.",
    "Nicks is ready.", "Hey, what time is it?",
]
