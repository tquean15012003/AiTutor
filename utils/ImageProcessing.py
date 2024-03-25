import base64

from fastapi import UploadFile, File

async def convert_image_to_base64(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise Exception(status_code=400, detail="Unsupported file type.")
    image_data = await image.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return image_base64