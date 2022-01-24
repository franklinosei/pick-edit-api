import torch
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageFile
from net import Net
from utils import DEVICE, test_transform
from typing import List
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
import PIL.Image as Image
import io
import io
from google.cloud.storage import client
from google.cloud import storage

Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = FastAPI()
client = storage.Client()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/styleimage/")
async def create_upload_files(files: List[UploadFile] = File(...)):

    try:
        bucket = client.get_bucket('pick-edit-storage')
        model = Net()
        model.eval()
        model = model.to(DEVICE)
        
        tf = test_transform()


        style_image = await files[0].read()
        content_image = await files[1].read()

        
        Ic = tf(Image.open(io.BytesIO(content_image))).to(DEVICE)
        Is = tf(Image.open(io.BytesIO(style_image))).to(DEVICE)

        Ic = Ic.unsqueeze(dim=0)
        Is = Is.unsqueeze(dim=0)
            
        with torch.no_grad():
            Ics = model(Ic, Is)

        name_cs = "ics.jpg"
        save_image(Ics[0], name_cs)

        # im = Image.fromarray(Ics[0].cpu().numpy())

        grid = make_grid(Ics[0])
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)

        b = io.BytesIO() 
        im.save(b, 'jpeg')
        im.close()

        # # Save the stylized image.
        blob = bucket.blob(f'styled/{files[1].filename}')
        blob.upload_from_string(b.getvalue(), content_type='image/jpeg')

        blob.make_public()

        return {"styled_image": FileResponse(blob.public_url)}
        
    except Exception as e:
        print(e)
        return {"error": str(e)}

