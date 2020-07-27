import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable

from torch import nn
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from permutator import Permutator


class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = torch.sigmoid(pred)

        return pred



class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size,args.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


# Add your own model here

class InteractE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations,
                 k_h: int = 10,
                 k_w: int = 5,
                 num_perm: int = 1,
                 inp_drop_p: float = 0.2,
                 hid_drop_p: float = 0.5,
                 feat_drop_p: float = 0.5,
                 kernel_size: int = 9,
                 num_filt_conv: int = 96,
                #  strategy: str = 'one_to_n',
                 neg_num: int = 0,
                 init_random = True):
        super(InteractE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_perm = num_perm					# number of permutation
        self.kernel_size = kernel_size
        
        self.entity_embeddings = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.relation_embeddings = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.loss = torch.nn.BCELoss()

        # Dropout regularization for input layer, hidden layer and embedding matrix
        self.inp_drop = nn.Dropout(inp_drop_p)
        self.hidden_drop = nn.Dropout(hid_drop_p)
        self.feature_map_drop = nn.Dropout2d(feat_drop_p)
        # Embedding matrix normalization
        self.bn0 = nn.BatchNorm2d(self.num_perm)

        self.k_h = k_h
        self.k_w = k_w
        flat_sz_h = k_h
        flat_sz_w = 2*k_w
        self.padding = 0

        # Conv layer normalization
        self.bn1 = nn.BatchNorm2d(num_filt_conv * self.num_perm)
        
        # Flattened embedding matrix size
        self.flat_sz = flat_sz_h * flat_sz_w * num_filt_conv * self.num_perm

        # Normalization
        self.bn2 = nn.BatchNorm1d(args.embedding_dim)

        # Matrix flattening
        self.fc = nn.Linear(self.flat_sz, args.embedding_dim)
        
        # Chequered permutation
        self.chequer_perm = Permutator(num_perm=self.num_perm, mtx_h=k_h, mtx_w=k_w).chequer_perm()

        # Bias definition
        self.register_parameter('bias', Parameter(torch.zeros(self.num_entities)))

        # Kernel filter definition
        self.num_filt_conv = num_filt_conv
        self.register_parameter('conv_filt', Parameter(torch.zeros(num_filt_conv, 1, kernel_size, kernel_size)))
        xavier_normal_(self.conv_filt)

    def init(self):
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, e1, rel):

        sub_emb	= self.entity_embeddings(e1)    # Embeds the subject tensor
        rel_emb	= self.relation_embeddings(rel)	# Embeds the relationship tensor
        
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        # self to access local variable.
        matrix_chequer_perm = comb_emb[:, self.chequer_perm]
        # matrix reshaped
        stack_inp = matrix_chequer_perm.reshape((-1, self.num_perm, 2*self.k_w, self.k_h))
        stack_inp = self.bn0(stack_inp)  # Normalizes
        x = self.inp_drop(stack_inp)	# Regularizes with dropout
        # Circular convolution
        x = self.circular_padding_chw(x, self.kernel_size//2)	# Defines the kernel for the circular conv
        x = F.conv2d(x, self.conv_filt.repeat(self.num_perm, 1, 1, 1), padding=self.padding, groups=self.num_perm) # Circular conv
        x = self.bn1(x)	# Normalizes
        x = F.relu(x)
        x = self.feature_map_drop(x)	# Regularizes with dropout
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)	# Regularizes with dropout
        x = self.bn2(x)	# Normalizes
        x = F.relu(x)
        
        # if self.strategy == 'one_to_n':
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1,0))
        x += self.bias.expand_as(x)
        # else:
        #     x = torch.mul(x.unsqueeze(1), self.entity_embeddings[self.neg_num]).sum(dim=-1)
        #     x += self.bias[self.neg_num]

        prediction = torch.sigmoid(x)

        return prediction
