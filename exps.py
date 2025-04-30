import argparse
import torch
import datetime
import json
import yaml
import os
from main_model import bay,metrla,pems03,pems04,pems07,pems08
from datasets import get_dataloader_bay,get_dataloader_Metrla,get_dataloader_Pems03,get_dataloader_Pems04,get_dataloader_Pems07,get_dataloader_Pems08
from utils import train, evaluate
from get_adj import get_adj_bay,get_adj_metrla,get_adj_pems03,get_adj_pems04,get_adj_pems07,get_adj_pems08
import torch

parser = argparse.ArgumentParser(description="Diff-Mxier")

parser.add_argument("--config", type=str, default="metrla.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.3)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--dataset", type=str, default="metrla")
parser.add_argument("--modelfolder", type=str, default="")

parser.add_argument("--nsample", type=int, default=100)

parser.add_argument("--block", action="store_true",help="Run or not.")


args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["train"]["block"] = args.block
config["train"]["seed"] = args.seed

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/"+args.dataset+"/"+args.dataset + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


if args.dataset == "bay":
    Model = bay
    dataloader = get_dataloader_bay
    adj = get_adj_bay()
elif args.dataset == "metrla":
    Model = metrla
    dataloader = get_dataloader_Metrla
    adj = get_adj_metrla()
elif args.dataset == "pems03":
    Model = pems03
    dataloader = get_dataloader_Pems03
    adj = get_adj_pems03()
elif args.dataset == "pems04":
    Model = pems04
    dataloader = get_dataloader_Pems04
    adj = get_adj_pems04()
elif args.dataset == "pems07":
    Model = pems07
    dataloader = get_dataloader_Pems07
    adj = get_adj_pems07()
elif args.dataset == "pems08":
    Model = pems08
    dataloader = get_dataloader_Pems08
    adj = get_adj_pems08()
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


train_loader, valid_loader, test_loader,mean,std = dataloader(config['diffusion'],
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    block=args.block
)

adj=torch.tensor(adj,dtype=torch.float32).to(args.device)
model = Model(config, device=args.device,adj=adj).to(args.device)
mean=torch.tensor(mean).to(args.device).float()
std=torch.tensor(std).to(args.device).float()

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/"+args.dataset+"/"+args.dataset+"_"+ args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=std, mean_scaler=mean, foldername=foldername)
