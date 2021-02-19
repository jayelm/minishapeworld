"""
Visualizing datasets with HTML
"""


import os

import numpy as np
from PIL import Image

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Shapeworld</title>
<style>
body {{ font-family: sans-serif; }}
img {{ padding: 10px; }}
img.yes {{ background-color: green; }}
img.no {{ background-color: red; }}
div.example {{ background-color: #eeeeee; }}
</style>
</head>
<body>
{}
</body>
</html>
"""


def make_example_html(example_i, labels, lang):
    return '<div class="example"><h1>{}</h1><p>{}</p></div>'.format(
        lang, make_image_html(example_i, labels)
    )


def make_image_html(example_i, labels):
    text_labels = ["no", "yes"]
    if len(labels.shape) > 0:
        return "".join(
            '<img src="{}_{}.png" class="{}">'.format(
                example_i, image_i, text_labels[label]
            )
            for image_i, label in enumerate(labels)
        )
    else:
        return '<img src="{}.png">'.format(example_i)


def visualize(img_dir, data, n=100):
    # Save to test directory
    data = {k: v[:n] for k, v in data.items()}
    data_arr = list(zip(data["imgs"], data["labels"], data["langs"]))
    for example_i, (example, labels, lang) in enumerate(data_arr):
        if len(example.shape) == 3:
            # Caption dataset
            example = np.transpose(example, (1, 2, 0))
            Image.fromarray(example).save(
                os.path.join(img_dir, "{}.png".format(example_i))
            )
        else:
            for image_i, image in enumerate(example):
                image = np.transpose(image, (1, 2, 0))
                Image.fromarray(image).save(
                    os.path.join(img_dir, "{}_{}.png".format(example_i, image_i))
                )

    example_html = [
        make_example_html(example_i, labels, lang)
        for example_i, (example, labels, lang) in enumerate(data_arr)
    ]
    index_fname = os.path.join(img_dir, "index.html")
    with open(index_fname, "w") as f:
        # Sorry for this code
        f.write(HTML_TEMPLATE.format("".join(example_html)))
