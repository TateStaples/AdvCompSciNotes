import real_time
from transformer import TransformerNet
import cv2
import discord
import torch

import aiofiles
import aiohttp


style = "udnie"
style_model = TransformerNet()
style_model.load_state_dict(torch.load(f"saved-models/{style}.pth"))

token = 'NzcwNzQwMDk3Njc4MDQ5Mjkw.X5h9pg.miBpFUHK9f_eRfkj0sU_sL5LX_M'
client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if len(message.attachments) == 0: return 
    url = message.attachments[0].url
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            f = await aiofiles.open('placeholder.png', mode='wb')
            await f.write(await response.read())
            await f.close()
    img = cv2.imread("placeholder.png")
    img = real_time.style_img(img, style_model)
    await send_img(message.channel, img)


async def send_img(channel, img):
    cv2.imwrite("placeholder.png", img)
    await channel.send(file=discord.File("placeholder.png"))

# video = cv2.VideoCapture(1)

if __name__ == '__main__':
    client.run(token)
    # _, frame = video.read()
    # print(frame.sum())
    # cv2.imwrite("source.png", frame)
    # cv2.imwrite("examples/source.jpg", frame)
    # img = cv2.imread("examples/dog.jpg")
    # new = real_time.style_img(img, style_model)
    # cv2.imwrite("test.png", new)