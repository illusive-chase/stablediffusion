import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
from tqdm import tqdm
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 图像加载函数 ===
# === 图像加载函数 ===
def load_image(path, size=512, center=True):
    img = Image.open(path).convert("RGB")
    top, left, height, width = 256, 512, size, size
    if center:
        transform = T.Compose([
            T.CenterCrop(size),      # 保证图像方形（裁剪短边）
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)     # [0,1] -> [-1,1]
        ])
    else:
        transform = T.Compose([
            # T.CenterCrop(size),      # 保证图像方形（裁剪短边）
            T.Lambda(lambda img: TF.crop(img, top=top, left=left, height=height, width=width)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)     # [0,1] -> [-1,1]
        ])
    img = transform(img)
    return img.unsqueeze(0).to(device)

def load_mask(path, size=512, center=True):
    mask = Image.open(path).convert("L")
    top, left, height, width = 256, 512, size, size
    if center:
        transform = T.Compose([
            T.CenterCrop(size),     # 保证图像方形
            T.ToTensor(),
            T.Lambda(lambda x: (x < 0.5).float())  # 二值化
        ])
    else:
        transform = T.Compose([
            # T.CenterCrop(size),     # 保证图像方形
            T.Lambda(lambda img: TF.crop(img, top=top, left=left, height=height, width=width)),
            T.ToTensor(),
            T.Lambda(lambda x: (x < 0.5).float())  # 二值化
        ])
    mask = transform(mask)
    return mask.unsqueeze(0).to(device)


def rsld(output_path, mask_path, image_path, y_path=None, gt_path = None, img_size=512, num_steps=1000, T=1000):
    # === 参数设置 ===
    prompt = ""
    step_size = 1e-6 # 更新步长sota
    coeff_similar = 0.000075 # x和z的似然度sota

    # step_size = 0
    # coeff_similar = 0 
    
    model_id = "/data/yuxuan/model/stable-diffusion-2-1"
    

    # === 加载模型 ===
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")

    log_dir = "runs/red_diff"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # 可以自定义 log_dir
    pipe=pipe.to('cuda')
    
    with torch.no_grad():
        unet, vae, scheduler = pipe.unet, pipe.vae, pipe.scheduler
        tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder

    shape_x = (1, 3, img_size, img_size)
    shape_z = (1, unet.config.in_channels, img_size // 8, img_size // 8)

    for param in vae.parameters():
        param.requires_grad = False

     # === 加载观测图像和mask ===
    cond_rgb = load_image(image_path, size=img_size, center=True)
    mask = load_mask(mask_path,size=img_size, center=True)
    y = cond_rgb * mask

    # === 编码文本提示 ===
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    z0 = torch.randn(shape_z, device=device, dtype=torch.float32, requires_grad=True)
    x0 = torch.randn(shape_x, device=device, dtype=torch.float32, requires_grad=True)
    optimizer_z = torch.optim.AdamW([z0], lr=0.8)
    optimizer_x = torch.optim.AdamW([x0], lr=0.4)

    for step in tqdm(range(num_steps)):
        optimizer_z.zero_grad()
        optimizer_x.zero_grad()
        # 采样一个时间步
        t = T - int(T / num_steps * step + 1)
        # 噪声 & forward
        noise = torch.randn_like(z0)
        alphas_cumprod = scheduler.alphas_cumprod.to(z0.device)
        alpha_prod_t = alphas_cumprod[t]
        alpha_t = torch.sqrt(alpha_prod_t)
        sigma_t = torch.sqrt(1 - alpha_prod_t)
        snr_t = sigma_t / alpha_t
        lamda_t = step_size * snr_t
        # 构造 z_t
        z_t = alpha_t * z0 + sigma_t * noise

        # optimize z
        # UNet 预测噪声
        noise_pred = unet(z_t, t, encoder_hidden_states=text_embeddings).sample
        # RED-diff loss: difference between predicted and true noise
        diff_noise = (noise_pred - noise).detach()  # detach 避免梯度传播
        # 解码当前 mu
        decoded = vae.decode(z0 / 0.18215).sample
        # 主要 loss
        loss_recon_z = torch.nn.functional.mse_loss(decoded , x0)
        # RED-Diff consistency term
        loss_red_z = lamda_t * (diff_noise * z0).mean()
        # 总 loss
        loss_z = loss_recon_z + loss_red_z
        writer.add_scalars(
            "Loss/z", {
            "total": loss_z.item(),
            "recon": loss_recon_z.item(),
            "red": loss_red_z.item()
        }, step)
        
        loss_z.backward()
        optimizer_z.step()

        # optimize x
        
        loss_recon_x = torch.nn.functional.mse_loss(x0*mask , y)
        decoded_z = vae.decode(z0 / 0.18215).sample
        loss_similar = torch.nn.functional.mse_loss(x0 , decoded_z)
        loss_x = loss_recon_x + coeff_similar * loss_similar
        writer.add_scalars("Loss/x", {
            "total": loss_x.item(),
            "recon": loss_recon_x.item(),
            "similar": loss_similar.item()
        }, step)
        loss_x.backward()
        optimizer_x.step()

        

    with torch.no_grad():
        # final_img = vae.decode(1 / 0.18215 * z0).sample
        final_img = x0
        print(final_img.min(), final_img.max())
        final_img = (final_img.clamp(-1, 1) + 1) / 2
        final_img = (final_img * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
        Image.fromarray(final_img).save(output_path)

        if y_path is not None:
            gt_y = ( y + 1 ) / 2
            gt_y = (gt_y * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
            Image.fromarray(gt_y).save(y_path)
        
        if gt_path is not None:
            gt = ( cond_rgb + 1 ) / 2
            gt = (gt * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
            Image.fromarray(gt).save(gt_path)

def predict_x0_from_noise(scheduler, latents, noise_pred, timestep):
    # 注意：timestep 是 tensor，例如 t=980，必须是 scheduler 中的元素之一
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(latents.device)
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)

    x0 = (latents - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
    return x0

def dps(psld=False):
    # === 参数设置 ===
    image_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_reprojected_full.png"  # 输入图像路径（RGB，值域[0,255]）
    mask_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_inpainting.png"         # mask路径（白色为可见，黑色为缺失）
    output_path = "/data/yuxuan/code/stablediffusion/result/dps_output.png"
    y_path = "/data/yuxuan/code/stablediffusion/result/dps_y.png"
    gt_path = "/data/yuxuan/code/stablediffusion/result/dps_gt.png"
    prompt = "a reasonable object with black background"
    num_inference_steps = 500
    step_size = 10 # DPS更新步长
    
    model_id = "/data/yuxuan/model/stable-diffusion-2-1"
    img_size=768


    # === 加载模型 ===
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to('cuda')
    with torch.no_grad():
        unet, vae, scheduler = pipe.unet, pipe.vae, pipe.scheduler
        tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder

    for param in vae.parameters():
        param.requires_grad = False

    # === 加载观测图像和mask ===
    cond_rgb = load_image(image_path, size=img_size)
    cond_rgb = vae.encode(cond_rgb).latent_dist.sample() * 0.18215
    cond_rgb = vae.decode(1 / 0.18215 * cond_rgb).sample
    mask = load_mask(mask_path,size=img_size)
    y = cond_rgb * mask
    print(y.min(), y.max())

    # print(mask.shape, mask.dtype, mask.min(), mask.max())  # 要求 float32, 范围[0,1]

    # === 编码文本提示 ===
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # === 初始化潜变量 ===
    z_t = torch.randn((1, unet.config.in_channels, img_size // 8, img_size // 8), device=device, dtype=torch.float16)
    # latents.requires_grad_()
    scheduler.set_timesteps(num_inference_steps, device=device)

    # === DPS采样过程 ===
    for t in tqdm(scheduler.timesteps):
       
        with torch.no_grad():
            z_t = scheduler.scale_model_input(z_t, t)
            noise_pred = unet(z_t, t, encoder_hidden_states=text_embeddings).sample
            zt_next = scheduler.step(noise_pred, t, z_t).prev_sample

        z_t = z_t.requires_grad_()

        pred_z0 = predict_x0_from_noise(scheduler=scheduler, latents=z_t, noise_pred=noise_pred, timestep=t)
        # 解码 & 计算已知区域损失
        decoded = vae.decode(1 / 0.18215 * pred_z0).sample
        loss_fn = torch.nn.MSELoss() 
        loss = loss_fn(decoded * mask, y)*2000
        print("Loss:", loss.item())
        grads = torch.autograd.grad(loss, z_t, retain_graph=True)[0]

        # print(torch.autograd.grad(loss, latents, retain_graph=True, allow_unused=True)[0])
        if t.item() > 800:
            zt_next = zt_next
        else: 
            zt_next = zt_next - step_size * grads
        
        if psld:
            encoded = vae.encode(decoded).latent_dist.sample() * 0.18215
            loss_regular = loss_fn(encoded, pred_z0)
            grads_regular = torch.autograd.grad(loss_regular, z_t, retain_graph=True)[0]
            print("Loss psld regular:", loss_regular.item())
            zt_next = zt_next - 0.1 * grads_regular
        
        print(torch.abs(grads).mean())
        print(torch.abs(grads_regular).mean())
        print(z_t.abs().mean())

        z_t = zt_next.detach_()


    # === 最终解码 ===
    with torch.no_grad():
        final_img = vae.decode(1 / 0.18215 * z_t).sample
        print(final_img.min(), final_img.max())
        final_img = (final_img.clamp(-1, 1) + 1) / 2
        final_img = (final_img * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()

        gt_y = ( y + 1 ) / 2
        gt_y = (gt_y * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()

        gt = ( cond_rgb + 1 ) / 2
        gt = (gt * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
        Image.fromarray(final_img).save(output_path)
        Image.fromarray(gt).save(gt_path)
        Image.fromarray(gt_y).save(y_path)
         # print(final_img.mean(), final_img.min(), final_img.max())
        

    print(f"dps完成，结果保存在 {output_path}")

def dps_new(output_path, mask_path, image_path, y_path=None, gt_path = None, img_size=512, num_steps=1000, T=1000, psld=True):
    # === 参数设置 ===
    prompt = ""
    num_inference_steps = 600
    step_size = 1200 # DPS更新步长
    
    model_id = "/data/yuxuan/model/stable-diffusion-2-1"
    img_size=512


    # === 加载模型 ===
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to('cuda')
    unet, vae, scheduler = pipe.unet, pipe.vae, pipe.scheduler
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder

    for param in vae.parameters():
        param.requires_grad = False

    # === 加载观测图像和mask ===
    cond_rgb = load_image(image_path, size=img_size)
    cond_rgb = vae.encode(cond_rgb).latent_dist.sample() * 0.18215
    cond_rgb = vae.decode(1 / 0.18215 * cond_rgb).sample
    mask = load_mask(mask_path,size=img_size)
    y = cond_rgb * mask
    print(y.min(), y.max())

    # print(mask.shape, mask.dtype, mask.min(), mask.max())  # 要求 float32, 范围[0,1]

    # === 编码文本提示 ===
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # === 初始化潜变量 ===
    z_t = torch.randn((1, unet.config.in_channels, img_size // 8, img_size // 8), device=device, dtype=torch.float32)
    # latents.requires_grad_()
    scheduler.set_timesteps(num_inference_steps, device=device)

    # === DPS采样过程 ===
    for t in tqdm(scheduler.timesteps):
       
        with torch.no_grad():
            z_t = scheduler.scale_model_input(z_t, t)
            noise_pred = unet(z_t, t, encoder_hidden_states=text_embeddings).sample
            zt_next = scheduler.step(noise_pred, t, z_t).prev_sample

        z_t = z_t.requires_grad_()

        pred_z0 = predict_x0_from_noise(scheduler=scheduler, latents=z_t, noise_pred=noise_pred, timestep=t)
        # 解码 & 计算已知区域损失
        decoded = vae.decode(1 / 0.18215 * pred_z0).sample
        loss_fn = torch.nn.MSELoss() 
        loss = loss_fn(decoded * mask, y)
        print("Loss:", loss.item())
        grads = torch.autograd.grad(loss, z_t, retain_graph=True)[0]

        # adapt_fun = torch.nn.L1Loss()
        lamda = step_size
        # lamda = (step_size / adapt_fun(decoded * mask, y)).detach()
        # print(f"lamda is {lamda}")
        # # print(torch.autograd.grad(loss, latents, retain_graph=True, allow_unused=True)[0])
        if t.item() > 750:
            zt_next = zt_next
        else: 
            zt_next = zt_next - lamda * grads

        
        
        if psld:
            encoded = vae.encode(decoded).latent_dist.sample() * 0.18215
            loss_regular = loss_fn(encoded, pred_z0)
            grads_regular = torch.autograd.grad(loss_regular, z_t, retain_graph=True)[0]
            print("Loss psld regular:", loss_regular.item())
            zt_next = zt_next - 0.01 * grads_regular
            print(torch.abs(grads_regular).mean())
        
        print(torch.abs(grads).mean())
        
        print(z_t.abs().mean())

        z_t = zt_next.detach_()

       # === 最终解码 ===
    with torch.no_grad():
        final_img = vae.decode(1 / 0.18215 * z_t).sample
        print(final_img.min(), final_img.max())
        final_img = (final_img.clamp(-1, 1) + 1) / 2
        final_img = (final_img * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
        Image.fromarray(final_img).save(output_path)

        if y_path is not None:
            gt_y = ( y + 1 ) / 2
            gt_y = (gt_y * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
            Image.fromarray(gt_y).save(y_path)
        
        if gt_path is not None:
            gt = ( cond_rgb + 1 ) / 2
            gt = (gt * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
            Image.fromarray(gt).save(gt_path)
        
        
        
        
    print(f"dps完成，结果保存在 {output_path}")


def red_diff(output_path=None, mask_path=None, image_path=None, y_path=None, gt_path = None, img_size=512, num_steps=1000, T=1000):
    # === 参数设置 ===
    image_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_reprojected_full.png"  # 输入图像路径（RGB，值域[0,255]）
    mask_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_inpainting.png"         # mask路径（白色为可见，黑色为缺失）
    output_path = "/data/yuxuan/code/stablediffusion/result/red_diff_output.png"
    y_path = "/data/yuxuan/code/stablediffusion/result/red_diff_y.png"
    gt_path = "/data/yuxuan/code/stablediffusion/result/red_diff_gt.png"
    prompt = ""
    step_size = 1e-7 # 更新步长

    model_id = "/data/yuxuan/model/stable-diffusion-2-1"
    img_size=512
    

    # === 加载模型 ===
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # writer = SummaryWriter(log_dir="runs/red_diff")  # 可以自定义 log_dir
    pipe=pipe.to('cuda')
    
    with torch.no_grad():
        unet, vae, scheduler = pipe.unet, pipe.vae, pipe.scheduler
        tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder


    shape_z = (1, unet.config.in_channels, img_size // 8, img_size // 8)

    for param in vae.parameters():
        param.requires_grad = False

     # === 加载观测图像和mask ===
    cond_rgb = load_image(image_path, size=img_size, center=True)
    mask = load_mask(mask_path,size=img_size, center=True)
    y = cond_rgb * mask

    # === 编码文本提示 ===
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    z0 = torch.randn(shape_z, device=device, dtype=torch.float32, requires_grad=True)
    optimizer_z = torch.optim.AdamW([z0], lr=0.1)

    for step in tqdm(range(num_steps)):
        optimizer_z.zero_grad()
        # 采样一个时间步
        t = T - int(T / num_steps * step + 1)
        # 噪声 & forward
        noise = torch.randn_like(z0)
        alphas_cumprod = scheduler.alphas_cumprod.to(z0.device)
        alpha_prod_t = alphas_cumprod[t]
        alpha_t = torch.sqrt(alpha_prod_t)
        sigma_t = torch.sqrt(1 - alpha_prod_t)
        snr_t = sigma_t / alpha_t
        lamda_t = step_size * snr_t
        # 构造 z_t
        z_t = alpha_t * z0 + sigma_t * noise

        # optimize z
        # UNet 预测噪声
        noise_pred = unet(z_t, t, encoder_hidden_states=text_embeddings).sample
        # RED-diff loss: difference between predicted and true noise
        diff_noise = (noise_pred - noise).detach()  # detach 避免梯度传播
        # 解码当前 mu
        decoded = vae.decode(z0 / 0.18215).sample
        # 主要 loss
        loss_recon_z = torch.nn.functional.mse_loss(decoded * mask , y)
        encoded = vae.encode(decoded).latent_dist.sample() * 0.18215
        loss_regular = 0 * torch.nn.functional.mse_loss(encoded, z0)
        # RED-Diff consistency term
        loss_red_z = lamda_t * (diff_noise * z0).mean()
        # 总 loss
        loss_z = loss_recon_z + loss_red_z + loss_regular
        # writer.add_scalars(
        #     "Loss/z", {
        #     "total": loss_z.item(),
        #     "recon": loss_recon_z.item(),
        #     "red": loss_red_z.item()
        # }, step)
        
        loss_z.backward()
        optimizer_z.step()


    with torch.no_grad():
        final_img = vae.decode(1 / 0.18215 * z0).sample
        print(final_img.min(), final_img.max())
        final_img = (final_img.clamp(-1, 1) + 1) / 2
        final_img = (final_img * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()

        gt_y = ( y + 1 ) / 2
        gt_y = (gt_y * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
        gt = ( cond_rgb + 1 ) / 2
        gt = (gt * 255).byte().permute(0, 2, 3, 1)[0].cpu().numpy()
        Image.fromarray(final_img).save(output_path)
        Image.fromarray(gt).save(gt_path)
        Image.fromarray(gt_y).save(y_path)


    

def inpainting():
    model_id = "/data/yuxuan/model/stable-diffusion-2-1-inpainting/"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,  # 或者 HuggingFace 模型名
        torch_dtype=torch.float16,
    ).to("cuda")
    # 载入图像和 mask
    img_size = 768

    image_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_reprojected.png"  # 输入图像路径（RGB，值域[0,255]）
    mask_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_inpainting.png"         # mask路径（白色为可见，黑色为缺失）
    init_image = load_image(image_path, size=img_size)
    mask = load_mask(mask_path,size=img_size)

    # pipe
    prompt = "an object with smooth color"
    num_inference_steps = 50
    guidance_scale = 7.5

    result = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    result.save("/data/yuxuan/code/stablediffusion/result/inpaint_result.png")
    print("Inpainting 完成，保存至 inpaint_result.png")


def img2img():
    model_id = "/data/yuxuan/model/stable-diffusion-2-1"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,  # 或者直接用 "stabilityai/stable-diffusion-2-1"
        torch_dtype=torch.float16,
    ).to("cuda")

    corrupt_img_path = "/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_gyx/gt_reprojected.png"
    output_path = "/data/yuxuan/code/stablediffusion/result/img2img_output.png"

    # 加载一张起始图像
    init_image = Image.open(corrupt_img_path).convert("RGB")
    init_image = init_image.resize((768, 768))  # SD-2.1 的推荐分辨率


    prompt = "a smooth colored object without changing much from given image"
    strength = 0.3  # 控制图像变化程度（1.0变化大，0.0无变化）
    guidance_scale = 7.5  # 文本引导强度

    output = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    ).images[0]

    output.save(output_path)
    print(f"img2img 完成，保存至 {output_path}")



def test():
    model_id = "/data/yuxuan/model/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
        
    image.save("/data/yuxuan/code/stablediffusion/result/astronaut_rides_horse.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', type=str, choices=["inpainting", "img2img","dps","psld", "rlsd", "red_diff"], default="rlsd")
    parser.add_argument('--view', type=int, default=20)
    parser.add_argument('--offset', type=int, default=40)
    parser.add_argument('--img_size', type=int, default=512)

    parser.add_argument("--image_path", type=str,
        default="/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_20_10/gt_reprojected_full.png",
        help="输入ground truth 图像路径")

    parser.add_argument("--mask_path", type=str,
        default="/data/yuxuan/code/RadianceFieldStudio/exports/hotdog_20_10/gt_inpainting.png",
        help="mask路径,白色为缺失,黑色为可见")

    parser.add_argument("--output_path", type=str,
        default="/data/yuxuan/code/stablediffusion/result/rlsd_20_10.png",
        help="最终输出图像的保存路径")

    parser.add_argument("--y_path", type=str,
        default="/data/yuxuan/code/stablediffusion/result/rlsd_y.png",
        help="观测y的保存路径")

    parser.add_argument("--gt_path", type=str,
        default="/data/yuxuan/code/stablediffusion/result/rlsd_gt.png",
        help="原始 ground truth 图像的保存路径")

    args = parser.parse_args()
    if args.mission == "inpainting":
        inpainting()
    elif args.mission == "img2img":
        img2img()
    elif args.mission == "dps":
        dps_new(image_path=args.image_path, mask_path=args.mask_path, 
                 output_path=args.output_path, y_path=args.y_path, gt_path=args.y_path, img_size=args.img_size, psld=False)
    elif args.mission == "psld":
        dps_new(psld=True)
    elif args.mission == "rlsd":
        rsld(image_path=args.image_path, mask_path=args.mask_path, 
                 output_path=args.output_path, y_path=args.y_path, gt_path=args.y_path, img_size=args.img_size)
    elif args.mission == "red_diff":
        red_diff()
    else:
        print("input wrong mission name")
    
