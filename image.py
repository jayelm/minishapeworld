"""
The image class which handles drawing/saving
"""

from PIL import Image
import aggdraw
import numpy as np
import constants as c


class IMG:
    def __init__(self):
        self.image = Image.new('RGB', (c.DIM, c.DIM))
        self.draw = aggdraw.Draw(self.image)

    def draw_shapes(self, shapes, flush=True):
        for shape_ in shapes:
            shape_.draw(self)
        if flush:
            self.draw.flush()

    def show(self):
        self.image.show()

    def array(self):
        return np.array(self.image, dtype=np.uint8)

    def float_array(self):
        return np.divide(np.array(self.image), 255.0)

    def save(self, path, filetype='PNG'):
        self.image.save(path, filetype)
