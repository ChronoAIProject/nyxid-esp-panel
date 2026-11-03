# Panel artwork

`nyxbot-default.jpg` is the persistent default page artwork, with the NyxID
N-mark and the default bot name **NyxBot** over the brand's cosmic background.
It is composed from existing brand assets; it is not a new AI illustration.
Regenerate with `tools/compose_nyxbot.py --font PATH_TO_BOLD_SANS_TTF`.
The checked-in render uses the local system's Arial Bold.

`nyxid-default.jpg` is the source background, an 800×1280 crop of NyxID's existing
`nyxid-brand-guide/public/photography/cosmic-1.png` (JPEG quality 90).
The final NyxBot JPEG is embedded in the application, so app-only updates need
no SPIFFS write. It remains on the live Panel page until new content arrives;
there is no splash-screen timer. No image API or API key was used.

Palette source: `nyxid-brand-guide/DESIGN_TOKENS.md`.

`nyx_panel_icon.c` contains the existing NyxID `nyxid-coloured-icon.png` N-mark,
resized onto a 112×112 black background and stored as RGB565 for the launcher.
`nyxid-mark.png` is that source N-mark, also used by the artwork compositor.
