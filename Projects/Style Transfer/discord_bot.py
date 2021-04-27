import real_time
from transformer import TransformerNet
import cv2
import discord
import torch

import aiofiles
import aiohttp


styles = ["udnie", "mosaic", "starry-night", "candy"]
models = dict()
for style in styles:
    model = TransformerNet()
    model.load_state_dict(torch.load(f"saved-models/{style}.pth"))
    models[style] = model

style = styles[0]
token = 'NzcwNzQwMDk3Njc4MDQ5Mjkw.X5h9pg.Mn1J3JfekkUTlhKctt66cGApDPs'
client = discord.Client()

@client.event
async def on_message(message):
    global style
    if message.author == client.user:
        return
    if len(message.attachments) == 0:
        style = message.content
        print(style)
        return
    url = message.attachments[0].url
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            f = await aiofiles.open('placeholder.png', mode='wb')
            await f.write(await response.read())
            await f.close()
    img = cv2.imread("placeholder.png")
    img = real_time.style_img(img, models[style])
    await send_img(message.channel, img)


async def send_img(channel, img):
    cv2.imwrite("placeholder.png", img)
    await channel.send(file=discord.File("placeholder.png"))

# video = cv2.VideoCapture(1)

if __name__ == '__main__':
    print("run")
    client.run(token)
    # _, frame = video.read()
    # print(frame.sum())
    # cv2.imwrite("source.png", frame)
    # cv2.imwrite("examples/source.jpg", frame)
    # img = cv2.imread("examples/dog.jpg")
    # new = real_time.style_img(img, style_model)
    # cv2.imwrite("test.png", new)