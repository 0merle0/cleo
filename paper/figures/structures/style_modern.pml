# Shared PyMOL style for every protein figure in this paper.
#   run/@ this file after loading a structure, then colour and render.
#
# NOTE: PyMOL splits commands on ';' even inside a comment, so no
# semicolons below -- a commented ';' makes the rest of the line run as a
# command and the include dies with a SyntaxError.
#
# Soft light, not outlines. An earlier version used ray_trace_mode 1, which
# draws a hard dark edge -- that matches figs/frag_animation.gif but reads as
# clip art at figure size. This instead lights the cartoon with ambient
# occlusion and a low-contrast key, so depth comes from shading and the
# fragment colours stay flat and legible.
#
# The cartoon is slightly inflated. Default PyMOL loops (0.2 radius) and sheets
# disappear at the sizes a journal column allows -- thickening them keeps the
# fold readable when the figure is 3 inches wide. Nothing here changes
# geometry, only how it is drawn.

bg_color white
set ray_opaque_background, 0
set orthoscopic, 1
set field_of_view, 20
set antialias, 2
set depth_cue, 0
set ray_shadows, 0

# --- soft lighting -------------------------------------------------------
set ray_trace_mode, 0
set ambient_occlusion_mode, 1
set ambient_occlusion_scale, 12
set ambient_occlusion_smooth, 12
# Ambient occlusion barely touches cartoons in PyMOL, so depth has to come
# from the key light. Ambient 0.20/direct 0.38 rendered almost perfectly flat --
# these values keep the soft palette legible without turning it into clip art.
set ambient, 0.13
set direct, 0.55
set reflect, 0.45
set light_count, 2
set specular, 0.22
set spec_power, 160
set spec_reflect, 0.08
set shininess, 35

# --- slightly inflated cartoon ------------------------------------------
set cartoon_sampling, 20
set cartoon_loop_radius, 0.36
set cartoon_tube_radius, 0.45
set cartoon_oval_length, 1.10
set cartoon_oval_width, 0.32
set cartoon_rect_length, 1.50
set cartoon_rect_width, 0.32
set cartoon_helix_radius, 2.60
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1
set cartoon_smooth_loops, 1
set cartoon_flat_sheets, 1
set cartoon_discrete_colors, 0
# -1 removes the grey helix interior, which reads as a second colour.
set cartoon_highlight_color, -1
