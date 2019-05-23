import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

class GraphAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_heads, dropout, alpha, last=False):
        super(GraphAttentionLayer1, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.last = last
        self.num_heads = num_heads

        
        self.weight = (nn.Parameter(torch.zeros(size=(num_heads, out_features, in_features))))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

        
        self.layer_att = (nn.Parameter(torch.zeros(size=(num_heads, 2*out_features, 1))))
        
 
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def Renorm(self, Adj):
        AdjTemp = Adj.clone().cuda()
        I = torch.FloatTensor(np.eye(Adj.size()[0])).cuda() 
    
        AdjTemp = AdjTemp + I
        Diag_IRT = torch.diag(torch.sum(AdjTemp, dim = 0)**(-0.5))
        AdjTemp = torch.mm(torch.mm(Diag_IRT, AdjTemp), Diag_IRT)

        return AdjTemp

    def forward(self, input, adj, thresh):
        
        #print (input.size())
        input = torch.mm(adj, input)
	
        aggregator = [ list() for i in range(input.size()[0])]

        for itr in range(self.num_heads):
             
            e = torch.zeros([input.size()[0],input.size()[0]]).cuda()
            
            attention = self.layer_att[itr].cuda()
            w = self.weight[itr].cuda()       

            for i in range(input.size()[0]):	
                for j in range(i+1, input.size()[0]):
                        
                        if(adj[i,j]>thresh): 
                            
                            concat = torch.cat([torch.mm(w, input[i][None,: ].permute(1,0)), torch.mm(w, input[j][None,: ].permute(1,0))]).permute(1,0)
                            
                            e[i,j] = torch.mm(concat,attention)
                            
                            e[j,i] = e[i,j]
                     

            alpha = F.softmax(e, dim = 0)

            temp = torch.zeros([self.out_features]).cuda()

            for i in range(input.size()[0]):
                for j in range(input.size()[0]):
                    print (adj[i,j], alpha[i,j])
                    temp += torch.sigmoid(torch.squeeze(alpha[i,j]*torch.mm(w, input[j][None,: ].permute(1,0))))
                
                aggregator[i].append(temp)


      
        if self.last != True:     
            h_prime = torch.reshape(torch.cat([torch.cat(aggregator[i]) for i in range(input.size()[0])]), [input.size()[0], self.num_heads*self.out_features]).cuda()
        else:
            
            h_prime = torch.reshape(torch.cat([torch.mean(torch.stack(aggregator[i]), dim = 0) for i in range(input.size()[0])]), [input.size()[0], self.out_features]).cuda()

        #print (h_prime.size())
        
        return h_prime


                 
            
        
        
        

        

                     

        
        




        






#-----------------------------------\
#----------------------------------------------------------------------------------------------------\

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def Renorm(self, Adj):
        AdjTemp = Adj.clone().cuda()
        I = torch.FloatTensor(np.eye(Adj.size()[0])).cuda() 
    
        AdjTemp = AdjTemp + I
        Diag_IRT = torch.diag(torch.sum(AdjTemp, dim = 0)**(-0.5))
        AdjTemp = torch.mm(torch.mm(Diag_IRT, AdjTemp), Diag_IRT)

        return AdjTemp

    def forward(self, input, adj):
        #print (torch.sum(input), torch.sum(self.W),'sum of input and w')
        h = torch.mm(input, self.W)
        N = h.size()[0]
        #print (h.size(),'This is sizeof h')
	#print (N,'This is N')

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
	#print (a_input.size(), 'This is a')
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
	#print (e.size(), 'size of e')

        zero_vec = torch.ones_like(e)
       
        #print (torch.sum(attention), 'Sum as e')
        attention = torch.where(adj > 0, e, zero_vec)
	
        #print (attention.size(), 'Size of attention')
        #print (torch.sum(attention),'Sum of attention')
        #attention = F.softmax(attention, dim=1)
        #print (torch.sum(attention),'Sum of attention after softm')
        attention = self.Renorm(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #print (torch.sum(attention),'Sum of attention after drop')
        #print (torch.sum(h),'Sum of h')
        h_prime = torch.matmul(attention, h)
        #print (torch.sum(h_prime),'Sum of h_prime')

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
