from PIL import Image
import base64
from io import BytesIO
from PIL import Image as PILImage

def image_to_base64(image: PILImage) -> str:
    """
    Convert an image to a base64 string.
    :param image:
    :return:
    """
    with BytesIO() as buffer:
        image.save(buffer, format=image.format or "PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

if __name__ == "__main__":
    image = Image.open("/Users/haruiz/Desktop/dog.jpg")
    image = image.resize((400,400))
    image.show()
    #print(image_to_base64(image))
