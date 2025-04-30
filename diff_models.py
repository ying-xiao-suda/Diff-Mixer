import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x


class channel_projection(nn.Module):
    '''
    B:batch
    C:channel
    L:length
    N:links/nodes

    in(B C N L)
    out(B gamaC N L)
    '''
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.proj=nn.Sequential(
            Rearrange('B C N L -> B L N C'),
            nn.Linear(dim,hidden_dim),
            Rearrange('B L N C -> B C N L')
         )
    def forward(self,x):
        x=self.proj(x)
        return x   


class t_mixer(nn.Module):
    '''
    B:batch
    C:channel
    L:length
    N:links/nodes

    in(B C N L)
    out(B C N L)
    '''
    def __init__(self,dim,hidden_dim,norm_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            FeedForward(dim,hidden_dim,dropout),
            norm(norm_dim),
         )
    def forward(self,x):
        x=self.net(x)+x
        return x

class AdjustedLinear(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.adj = adj  

    def forward(self, x):
      
        adj_weight = torch.matmul(self.adj, self.linear.weight.t())  
        output = torch.matmul(x, adj_weight) + self.linear.bias 
        return output

class s_mixer(nn.Module):
    '''
    B:batch
    C:channel
    L:length
    N:links/nodes

    in(B C N L)
    out(B C N L)
    '''
    def __init__(self, dim, hidden_dim, adj, dropout=0.):
        super().__init__()
        self.layer_normal = nn.LayerNorm(dim)
        self.adj = adj 
        self.net1 = nn.Sequential(
            AdjustedLinear(dim, hidden_dim, adj), 
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.net2 = nn.Sequential(
            AdjustedLinear(hidden_dim, dim, adj),  
            nn.Dropout(dropout)
        )

    def forward(self, x):
        res = x
        x = rearrange(x, 'B C N L -> B C L N') 

        x = self.net1(x)

        x = self.net2(x)
        x = self.layer_normal(x)
        x = rearrange(x, 'B C L N -> B C N L') + res  
        return x

class c_mixer(nn.Module):
    '''
    B:batch
    C:channel
    L:length
    N:links/nodes

    in(B C N L)
    out(B gamaC N L)
    '''
    def __init__(self,dim,hidden_dim,norm_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            Rearrange('B C N L -> B L N C'),
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Rearrange('B L N C-> B C N L')
         )
    def forward(self,x):
        x=self.net(x)
        return x    

class norm(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(
            Rearrange('B C N L -> B C L N'),
            nn.LayerNorm(dim),
            Rearrange('B C L N -> B C N L')
         )
    def forward(self,x):
        x=self.net(x)
        return x    


class encoder(nn.Module):
    def __init__(self, dim, sequence_len, sequence_hid, dim_hid, channel_list, adj, dropout=0.):
        super().__init__()

        self.t_mixer_shared = t_mixer(sequence_len, sequence_hid, dim, dropout)
        self.s_mixer_shared = s_mixer(dim, dim_hid, adj, dropout)
        

        self.c_mixers = nn.ModuleList([])
        for i in range(len(channel_list) - 1):
            c_in = channel_list[i]
            c_out = channel_list[i+1]
            self.c_mixers.append(c_mixer(c_in, c_out, dim, dropout))
    
    def forward(self, x):

        for c_mixer in self.c_mixers:
            x = self.t_mixer_shared(x)
            x = self.s_mixer_shared(x)
            x = c_mixer(x)
        return x

class decoder(nn.Module):
    def __init__(self, dim, sequence_len, sequence_hid, dim_hid, channel_list, adj, dropout=0.):
        super().__init__()

        self.t_mixer_shared = t_mixer(sequence_len, sequence_hid, dim, dropout)
        self.s_mixer_shared = s_mixer(dim, dim_hid, adj, dropout)
        

        self.c_mixers = nn.ModuleList([])
        list_len = len(channel_list)
        for i in range(len(channel_list) - 1):
            c_in = channel_list[list_len - i - 1]
            c_out = channel_list[list_len - i - 2]
            self.c_mixers.append(c_mixer(c_in, c_out, dim, dropout))
    
    def forward(self, x):
        
        for c_mixer in self.c_mixers:
            x = self.t_mixer_shared(x)
            x = self.s_mixer_shared(x)
            x = c_mixer(x)
        return x  




# Diffusion Embedding
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        #读取table的第dilffusion_step行
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x
    
    #positional encoding (T,dim*2)
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table




class diff_model(nn.Module):
    def __init__(self, config, inputdim,adj):
        super().__init__()
        self.channels = config["channels"]
        self.gama=config["gama"]
        self.layers=config["layers"]
        self.channel_list=[]
        self.channel_list.append(self.channels)
        c_temp=self.channels
        for _ in range(self.layers):
            c_temp=int(c_temp*self.gama)
            self.channel_list.append(c_temp)
        
        print("Channel List:",self.channel_list)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        
        self.diffusion_projection1 = nn.Linear(config["diffusion_embedding_dim"],self.channel_list[0])
        self.cond_projection1 = channel_projection(config["side_dim"],self.channel_list[0])
        
        self.diffusion_projection2 = nn.Linear(config["diffusion_embedding_dim"],self.channel_list[-1])
        self.cond_projection2 = channel_projection(config["side_dim"],self.channel_list[-1])
        
        self.input_embdding = channel_projection(inputdim,self.channel_list[0])
        self.encoder_layer=encoder(
                                   dim=config['num_of_vertices'],
                                   sequence_len=config['seq_len'],
                                   sequence_hid=config['seq_hid'],
                                   dim_hid=config['n_hid'],
                                   channel_list=self.channel_list,
                                   adj=adj,
                                   dropout=config['dropout']
                                  )
        self.decoder_layer=decoder(
                                   dim=config['num_of_vertices'],
                                   sequence_len=config['seq_len'],
                                   sequence_hid=config['seq_hid'],
                                   dim_hid=config['n_hid'],
                                   channel_list=self.channel_list,
                                   adj=adj,
                                   dropout=config['dropout']
                                  )

       
        self.out=channel_projection(self.channel_list[0],1)


    #这里的cond_info指的是side_info
    def forward(self, x, cond_info,diffusion_step):
        B, _, K, L = x.shape
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        db,_=diffusion_emb.shape
        diffusion_emb1 = self.diffusion_projection1(diffusion_emb).unsqueeze(-1).unsqueeze(-1).repeat(B//db,1, K, L)
        diffusion_emb2 = self.diffusion_projection2(diffusion_emb).unsqueeze(-1).unsqueeze(-1).repeat(B//db,1, K, L)

        cond_info1 = self.cond_projection1(cond_info).reshape(B, -1, K , L)
        cond_info2 = self.cond_projection2(cond_info).reshape(B, -1, K , L)
        
        
        x=self.input_embdding(x)#(B,channels, K , L)
        
        x=self.encoder_layer(x + cond_info1 + diffusion_emb1)
        
        x=self.decoder_layer(x + cond_info2 + diffusion_emb2)
        
        x =self.out(x).reshape(B,K,L)

        return x

