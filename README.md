# minishapeworld

This is a super-lightweight version of ShapeWorld.

Currently supported datasets:

- Concept (multiple targets, multiple distractors, one caption)
- Reference (single target, multiple distractors, one caption)
- Caption (single target, one caption)

Crucially, datasets support "hard examples": i.e. if the
target caption is "a red shape above a square", distractor images will include
concepts like "a red shape above a circle", "a blue shape above a square", and
"a red shape BELOW a square".

The kinds of images (and captions) generated can be specified with the
`--img_type` option:

- `single`: Single existential captions (e.g. "blue square", "square", "blue
    shape")
- `spatial`: Spatial captions (e.g. "blue square", "square", "blue
    shape")

For spatial captions only (for now) you can specify the number of distractor shapes in each image with the `--n_distractors` option, specifying either a single number or two (`min`, `max`).

## Dependencies

- Python 3.6+
- aggdraw
- shapely
