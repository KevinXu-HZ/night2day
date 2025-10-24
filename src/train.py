import os, itertools, random, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.unpaired_dataset import UnpairedImageDataset
from src.models.networks import ResnetGenerator, NLayerDiscriminator
from src.losses.losses import GANLoss, SemanticConsistencyLoss
import torchvision.utils as vutils

def denorm(x): return (x.clamp(-1,1)+1)/2

def main(
    data_root=r"D:\datasets\dark_zurich",
    domain_a="night",
    domain_b="day",
    image_size=512,
    batch_size=4,
    max_epochs=20,
    lr=2e-4,
    lambda_cyc=10.0,
    lambda_id=0.5,
    lambda_sem=1.0,
    num_workers=0,
    log_dir="experiments/run1",
    save_every=5
):
    os.makedirs(log_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    ds = UnpairedImageDataset(root=data_root, domain_a=domain_a, domain_b=domain_b,
                              image_size=image_size, train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    G_A2B, G_B2A = ResnetGenerator().to(device), ResnetGenerator().to(device)
    D_A, D_B = NLayerDiscriminator().to(device), NLayerDiscriminator().to(device)

    gan, l1 = GANLoss().to(device), nn.L1Loss()
    sem = SemanticConsistencyLoss(device=device)

    opt_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr, betas=(0.5,0.999))
    opt_D = optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=lr, betas=(0.5,0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    step = 0
    for epoch in range(1, max_epochs+1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{max_epochs}")
        for batch in pbar:
            A, B = batch["A"].to(device), batch["B"].to(device)

            # ---- G ----
            opt_G.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                fake_B, rec_A = G_A2B(A), G_B2A(G_A2B(A))
                fake_A, rec_B = G_B2A(B), G_A2B(G_B2A(B))
                idt_B, idt_A = G_A2B(B), G_B2A(A)

                loss_gan = gan(D_B(fake_B), True) + gan(D_A(fake_A), True)
                loss_cyc = l1(rec_A, A)*lambda_cyc + l1(rec_B, B)*lambda_cyc
                loss_id  = (l1(idt_A, A) + l1(idt_B, B))*lambda_cyc*lambda_id
                loss_sem = (sem(A, fake_B) + sem(B, fake_A))*lambda_sem
                loss_G = loss_gan + loss_cyc + loss_id + loss_sem

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ---- D ----
            opt_D.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss_D_A = 0.5*(gan(D_A(A), True) + gan(D_A(fake_A.detach()), False))
                loss_D_B = 0.5*(gan(D_B(B), True) + gan(D_B(fake_B.detach()), False))
                loss_D = loss_D_A + loss_D_B
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

            step += 1
            pbar.set_postfix(G=f"{loss_G.item():.3f}", D=f"{loss_D.item():.3f}", cyc=f"{loss_cyc.item():.3f}", sem=f"{loss_sem.item():.3f}")

            if step % 500 == 0:
                os.makedirs(f"{log_dir}/samples", exist_ok=True)
                import torch as _t
                grid = _t.cat([A[:1], fake_B[:1], rec_A[:1], B[:1], fake_A[:1], rec_B[:1]], dim=0)
                vutils.save_image(denorm(grid), f"{log_dir}/samples/step_{step:07d}.png", nrow=3)

        if epoch % save_every == 0:
            os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
            import torch as _t
            _t.save(G_A2B.state_dict(), f"{log_dir}/checkpoints/G_A2B_e{epoch}.pt")
            _t.save(G_B2A.state_dict(), f"{log_dir}/checkpoints/G_B2A_e{epoch}.pt")
            _t.save(D_A.state_dict(),    f"{log_dir}/checkpoints/D_A_e{epoch}.pt")
            _t.save(D_B.state_dict(),    f"{log_dir}/checkpoints/D_B_e{epoch}.pt")

if __name__ == "__main__":
    main()
