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
#   outline weight   ray_trace_gain 0.1 and 1.0 are indistinguishable in EDGE
#                    WIDTH -- width follows resolution, not gain. But gain also
#                    darkens interior creases, and that part is NOT cosmetic
#                    (see lighting).
#   trace mode       0 has no edges and reads weakest. 3 QUANTISES COLOUR --
#                    orange went fluorescent -- which is disqualifying when
#                    colour carries fragment identity. 1 is the one to use.
#                    2 is line art with colour stripped, useful for a ghosted
#                    structure (see GHOST below) and nothing else.
#   lighting         WRONG in the first sweep. `reflect_power 0` makes diffuse
#                    falloff perfectly flat, so every ribbon rendered as one
#                    solid fill and the fold had no form at all -- a helix read
#                    as a flat zigzag, not a tube. That is invisible at the
#                    thumbnail size the sweep was judged at and obvious at
#                    print size. reflect_power is back at PyMOL's default 1,
#                    with ambient down and direct up to keep the midtones from
#                    washing out. Cast shadows (ray_shadow on, shadow_mode 2)
#                    were tried with it and rejected: they muddy whichever
#                    colour sits under another, and green took it worst.
#                    Ambient occlusion and a specular highlight were also tried
#                    -- AO barely moves cartoons, and specular puts a gloss on
#                    the light colours that fights fragment identity.
#   cartoon geometry THE setting that matters. Inflating loops is what keeps
#                    the fold legible at column width.
#   transparency     0.25 and 0.45 both wash out badly on white. Opaque.

bg_color white
set ray_opaque_background, 1
set orthoscopic, 1
# antialias 3, not 2. Edge quality on the ray-traced outline is set here,
# not by output resolution -- a soft-looking hairline is undersampled, not
# under-resolved.
set antialias, 3
set depth_cue, 0
set ray_shadow, off

# --- outlined, flat-lit ---------------------------------------------------
set ray_trace_mode, 1
set ray_trace_color, black
# Gain darkens interior creases as well as drawing the outline. 0.1 gave a
# clean edge around a shape with nothing inside it.
set ray_trace_gain, 0.45
set ray_trace_disco_factor, 1
set specular, 0
# These four are one setting. reflect_power 0 (what this file used to say) is
# flat shading -- do not set it back without re-reading the lighting note above.
set ambient, 0.28
set direct, 0.58
set reflect, 1
set reflect_power, 1

# --- cartoon geometry -----------------------------------------------------
set cartoon_sampling, 12
set cartoon_transparency, 0
# MATCH THE LOOP TO THE RIBBON. cartoon_loop_radius is a RADIUS while
# cartoon_oval_width is a full thickness, so a loop radius of r meets a ribbon
# of thickness w as 2r against w. At loop_radius 0.55 against oval_width 0.40
# the loop was 2.75x thicker than the ribbon, and every loop-to-helix junction
# stepped visibly. Half of oval_width is the match -- 0.20 here, which is also
# PyMOL's default. 0.25 keeps the loops a touch more present without stepping.
# Raise these two together or not at all.
set cartoon_loop_radius, 0.25
set cartoon_loop_quality, 12
# Loops are inflated, helices and strands are NOT. Widening them was tried at
# two levels and reverted: it thickens the fold without making it clearer, and
# on a barrel the helices start to close up the interior. Only the loops
# needed help -- at default they go to hairlines at column width.
set cartoon_oval_width, 0.40
set cartoon_rect_width, 0.40
# fancy_helices OFF. It gives the helix ribbon a rounded, raised edge --
# with reflect_power back at 1 the shading already carries that form, and
# the two together read as over-modelled.
set cartoon_fancy_helices, 0
# cartoon_dumbbell_radius is left at its default. Rounding the ribbon edges
# was swept at 0.26 / 0.40 / 0.55 and reverted -- PyMOL gives little visible
# change on strands (cartoon_fancy_sheets barely moves them), so the effect
# lands on helices only and the two stop matching. Widening helices and
# strands was tried and reverted for the same reason. Only the loops are
# adjusted, below, because at default they go to hairlines at column width.
set cartoon_flat_sheets, off
set cartoon_smooth_loops, 1
set cartoon_gap_cutoff, 0
