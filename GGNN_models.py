import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from get_yolo_scores import DataFeeder

class GGNN(nn.Module):

    def __init__(self, nodes, hidden_size, num_outputs, edge_types):
        super(GGNN, self).__init__()
        self.nodes = nodes
        self.hidden_size = hidden_size
        self.edge_types = edge_types

        # self.Wadj = torch.Tensor(self.edge_types)
        self.Wadj_out = nn.Parameter(torch.zeros(self.edge_types))
        nn.init.normal_(self.Wadj_out, mean=0, std=1)
        self.Wadj_out.requires_grad_(True)

        self.Wadj_in = nn.Parameter(torch.zeros(self.edge_types))
        nn.init.normal_(self.Wadj_in, mean=0, std=1)
        self.Wadj_in.requires_grad_(True)

        self.bias_in = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_in.requires_grad_(True)

        self.bias_out = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_out.requires_grad_(True)

        # self.grucell = nn.GRUCell(hidden_size, hidden_size)
        self.r1 = nn.Linear(2*hidden_size, hidden_size) 
        self.r2 = nn.Linear(hidden_size, hidden_size) 
        self.reset_gate = nn.Sigmoid()

        self.z1 = nn.Linear(2*hidden_size, hidden_size) 
        self.z2 = nn.Linear(hidden_size, hidden_size) 
        self.update_gate = nn.Sigmoid()

        self.t1 = nn.Linear(2*hidden_size, hidden_size) 
        self.t2 = nn.Linear(hidden_size, hidden_size) 
        self.transform = nn.Tanh()

        # self.imp_fc = nn.Sequential(
        #     nn.Linear(nodes*hidden_size, nodes),
        #     nn.Sigmoid()
        #     )

        self.out_fc = nn.Sequential(
            nn.Linear(nodes*hidden_size, num_outputs),
            nn.Sigmoid()
            )

       

    def forward(self, hidden_states, adj_matrix_out, adj_matrix_in):
        
        wadj_v_o = self.Wadj_out.unsqueeze(1).expand(self.nodes, self.edge_types , 1)
        wadj_v_i = self.Wadj_in.unsqueeze(1).expand(self.nodes, self.edge_types , 1)


        #print (wadj_v_o.shape, adj_matrix_out.shape)
        adj_matrix_1 = torch.bmm(adj_matrix_out, wadj_v_o).squeeze()
        adj_matrix_2 = torch.bmm(adj_matrix_in, wadj_v_i).squeeze()
        # print(adj_matrix, adj_matrix.requires_grad)
        t=0
        while t<3:
            
            actv = torch.matmul(adj_matrix_1, hidden_states) + self.bias_out
            actv_in = torch.matmul(adj_matrix_2, hidden_states) + self.bias_in
            actv = torch.cat((actv, actv_in), 2)
            r1 = self.r1(actv)
            r2 = self.r2(hidden_states)
            r = self.reset_gate(r1+r2)

            z1 = self.z1(actv)
            z2 = self.z2(hidden_states)
            z = self.update_gate(z1+z2)

            t1 = self.t1(actv)
            t2 = self.t2(r*hidden_states)
            h_cap = self.transform(t1+t2)

            hidden_states = (1 - z) * hidden_states + z*h_cap
            t+=1
        hidden_states_view = hidden_states.view(hidden_states.size()[0], -1)

        output = self.out_fc(hidden_states_view)

        return output
        # return adj_matrix
