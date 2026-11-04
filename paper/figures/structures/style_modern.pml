# Shared PyMOL style for every protein figure in this paper.
#   run/@ this file after loading a structure, then colour and render.
#
# TWO TRAPS, both of which cost real time before they were found.
#
# 1. PyMOL splits commands on a semicolon EVEN INSIDE A COMMENT, so a
#    commented ';' runs the rest of the line as a command and the include dies
#    with a SyntaxError. There are no semicolons below.
#
# 2. `pymol -cq` reads ~/.pymolrc automatically. A personal rc then applies to
#    every render -- the one on this machine sets cartoon_transparency 0.45 --
#    so figures silently depend on whose account built them. render.py invokes
#    `pymol -cqk` to suppress it. Everything this file needs is stated here.
#
# Chosen from a 20-setting sweep rendered side by side (outline weight,
# ray_trace_mode, lighting, cartoon geometry, transparency). What that showed:
#
#   outline weight   ray_trace_gain 0.1 and 1.0 are indistinguishable. Edge
#                    width follows resolution, not gain, so the low value is
#                    kept and nothing is bought by raising it.
#   trace mode       0 has no edges and reads weakest. 3 QUANTISES COLOUR --
#                    orange went fluorescent -- which is disqualifying when
#                    colour carries fragment identity. 1 is the one to use.
#                    2 is line art with colour stripped, useful for a ghosted
#                    structure (see GHOST below) and nothing else.
#   lighting         flat / modelled / glossy / matte are nearly the same once
#                    outlines are on. Not worth tuning.
#   cartoon geometry THE setting that matters. Inflating loops is what keeps
#                    the fold legible at column width.
#   transparency     0.25 and 0.45 both wash out badly on white. Opaque.

bg_color white
set ray_opaque_background, 1
set orthoscopic, 1
set antialias, 2
set depth_cue, 0
set ray_shadow, off

# --- outlined, flat-lit ---------------------------------------------------
set ray_trace_mode, 1
set ray_trace_color, black
set ray_trace_gain, 0.1
set ray_trace_disco_factor, 1
set specular, 0
set ambient, 0.40
set direct, 0.45
set reflect, 1
set reflect_power, 0

# --- cartoon geometry -----------------------------------------------------
set cartoon_sampling, 12
set cartoon_transparency, 0
set cartoon_loop_radius, 0.55
# Loops are inflated, helices and strands are NOT. Widening them was tried at
# two levels and reverted: it thickens the fold without making it clearer, and
# on a barrel the helices start to close up the interior. Only the loops
# needed help -- at default they go to hairlines at column width.
set cartoon_oval_width, 0.40
set cartoon_rect_width, 0.40
set cartoon_fancy_helices, 1
# Rounded ribbon edges -- the "partially inflated" look. 0.16 is the
# default and reads as flat tape. Swept 0.26 / 0.40 / 0.45: 0.40 is where
# the edge is visibly round without the helix losing its coil.
set cartoon_dumbbell_radius, 0.40
set cartoon_flat_sheets, off
set cartoon_smooth_loops, 1
set cartoon_gap_cutoff, 0
