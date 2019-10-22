# minishapeworld

This is a super-lightweight version of ShapeWorld.

Currently supported datasets:

- Concept (multiple targets, multiple distractors, one caption)
- Reference (single target, multiple distractors, one caption)
- Caption (single target, one caption)

Crucially, concept/reference dataseets support "hard examples": i.e. if the
target caption is "a red shape above a square", distractor images will incldue
concepts like "a red shape above a circle", "a blue shape above a circle", and
"a red shape BELOW a circle".

The kinds of images (and captions) generated can be specified with the
`--img_type` option:

- `single`: Single existential captions (e.g. "blue square", "square", "blue
    shape")
- `spatial`: Spatial captions (e.g. "blue square", "square", "blue
    shape")

## Dependencies

- Python 3.6+
- aggdraw
- shapely
