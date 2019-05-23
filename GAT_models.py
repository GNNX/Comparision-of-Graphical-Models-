import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_layers import GraphAttentionLayer, SpGraphAttentionLayer

class GAT1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT1, self).__init__()
        self.dropout = dropout

        self.gat1 = GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha)
        self.gat2 = GraphAttentionLayer(nhid*nheads, nclass, nheads, dropout, alpha, last=True)


    def forward(self, x, adj, thresh):
        print (torch.sum(x))
        x = F.relu(F.dropout(self.gat1(x, adj, thresh), self.dropout, training = self.training))
        print (torch.sum(x))
        x = F.relu(F.dropout(self.gat2(x, adj, thresh), self.dropout, training = self.training))
        print (torch.sum(x))
       
        return x



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        #print (x.size(),'Size of x')
        #print (torch.sum(x), 'Sum of x')
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        #print (x.size(),'Size of x after 1')
        #print (torch.sum(x),'Sum of x after 1')

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))

        #x = F.log_softmax(x, dim = 1)

        #print (x.size(),'Size of x after 2')
        #print (torch.sum(x),'Sum of x after 2')
       
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

