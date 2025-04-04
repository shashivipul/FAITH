import copy
import torch
import numpy as np
# from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix
from utils import *
from utils_guide import *
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Training for teacher GNNs
def train(model, g, feats, labels, criterion, optimizer, adj, idx):

    model.train()

    logits = model(g, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx], labels[idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Testing for teacher GNNs
def evaluate(model, g, feats):

    model.eval()

    with torch.no_grad():
        logits = model(g, feats)
        out = logits.log_softmax(dim=1)

    return logits, out
  

# Training for student MLPs
def train_mini_batch(model, edge_idx, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx, adj, Delta,lap_if,node_energy_teacher, param):

    model.train()

    logits= model(None, feats)
    out = logits.log_softmax(dim=1)
    loss_l = criterion_l(out[idx], labels[idx])
    loss_t = criterion_t((logits/param['tau']).log_softmax(dim=1), (out_t_all/param['tau']).log_softmax(dim=1))

    values_delta = torch.tensor(Delta.data, dtype=torch.float32, device='cuda')
    indices_delta = torch.tensor([Delta.row, Delta.col], dtype=torch.int64, device='cuda')
    Delta_sparse = torch.sparse_coo_tensor(indices_delta, values_delta, size=Delta.shape, dtype=torch.float32)
    smooth_emb = torch.sparse.mm(Delta_sparse, logits)
    row_l2_norms = torch.norm(smooth_emb, p=2, dim=1)
    l21_norm = torch.norm(row_l2_norms, p=1)
    loss_proposed_unfairness = l21_norm
  
    adj_sparse = torch.tensor(adj, dtype=torch.float32).to_sparse()
    loss_energy = compute_loss_energy(adj_sparse, logits, node_energy_teacher, edge_idx, batch_size=500000)
  
    loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb']) + loss_energy * beta + gamma* loss_proposed_unfairness
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step() 
  
    return loss_l.item() * param['lamb'], loss_t.item() * (1-param['lamb']) + loss_energy.item() *beta + gamma* loss_proposed_unfairness.item()


# Testing for student MLPs
def evaluate_mini_batch(model, feats):

    model.eval()

    with torch.no_grad():
        logits = model(None, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


def get_IF(logits,feats, adj):
    adj_sim = csr_matrix(adj)
    sim_if = calculate_similarity_matrix(adj_sim, feats, metric='cosine')
    lap_if = laplacian(sim_if)
    lap_if = convert_sparse_matrix_to_sparse_tensor(lap_if).to(logits.device)
    individual_fairness = torch.trace(torch.mm(logits.t(), torch.sparse.mm(lap_if,logits))).item()
    return individual_fairness/1000  

def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
      
        adj_1 = g.adjacency_matrix()                       
        row_1 = adj_1.indices()[0].numpy()
        col_1 = adj_1.indices()[1].numpy()
        values_1 = adj_1.val
        adj_mat_1 = coo_matrix((values_1,(row_1,col_1)),shape=adj_1.shape)
        adj_mat_final = adj_mat_1.toarray()   
        adj_mat_final_t = torch.tensor(adj_mat_final, dtype=torch.float32)
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_g = g.subgraph(idx_obs).to(device)

    g = g.to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            train_loss = train(model, g, feats, labels, criterion, optimizer,adj_mat_final, idx_train)
            logits_e, out = evaluate(model, g, feats)
                 
            IF = get_IF(logits_e,feats, adj_mat_final)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])
        else:
            train_loss = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            _, obs_out = evaluate(model, obs_g, obs_feats)
            train_acc = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            val_acc = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
            _, out = evaluate(model, g, feats)
            test_acc = evaluator(out[idx_test_ind], labels[idx_test_ind])

        if epoch % 1 == 0:

            print("\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | IF: {:.4f}\033[0m".format(
    epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best, IF))


        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            best_IF = IF
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break

    model.load_state_dict(state)
    model.eval()
    print(f"Best Validation Accuracy: {val_best:.4f}, Test Accuracy: {test_val:.4f}, IF: {best_IF:.4f}")

    if param['exp_setting'] == 'tran':
        out, _ = evaluate(model, g, feats)    
        device = out.to(device)
        adj_mat_final_t = adj_mat_final_t.to(device)
        row, col = torch.nonzero(adj_mat_final_t, as_tuple=True)  
        values = adj_mat_final_t[row, col]  
        diff_teacher = out[row] - out[col]  
        squared_diff_teacher = torch.sum(diff_teacher**2, dim=1)  
        weighted_diff_teacher = values * squared_diff_teacher  
        node_energy_teacher = torch.zeros(out.size(0)).to(device)
        node_energy_teacher = node_energy_teacher.index_add(0, row, weighted_diff_teacher)

    else:
        obs_out, _ = evaluate(model, obs_g, obs_feats)
        out, _ = evaluate(model, g, feats)
        out[idx_obs] = obs_out
    return out, test_acc, test_val, test_best, state,node_energy_teacher,best_IF


def train_student(param, model, g, feats, labels, out_t_all, indices, criterion_l, criterion_t, evaluator, optimizer,node_energy_teacher,KD_model):

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
        num_node = feats.shape[0]
        edge_idx_list = extract_indices(g)
      
        adj_1 = g.adjacency_matrix()                       
        row_1 = adj_1.indices()[0].numpy()
        col_1 = adj_1.indices()[1].numpy()
        values_1 = adj_1.val
        adj_mat_1 = coo_matrix((values_1,(row_1,col_1)),shape=adj_1.shape)
        adj_mat_final_s = adj_mat_1.toarray()    
        adj_mat_final_st = torch.tensor(adj_mat_final_s, dtype=torch.float32).to(out_t_all.device)
        adj_sim = csr_matrix(adj_mat_final_s)
        sim_if = calculate_similarity_matrix(adj_sim, feats, metric='cosine')
        Delta = GDO_optimized(sim_if)
        lap_if = laplacian(sim_if)

    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_out_t = out_t_all[idx_obs]
        num_node = obs_feats.shape[0]
        edge_idx_list = extract_indices(g.subgraph(idx_obs))

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    print('Student')

    for epoch in range(1, param["max_epoch"] + 1):

        if epoch == 1:
            edge_idx = edge_idx_list[0]   

        elif (epoch >= 50 and epoch % 10 == 0 and param['dataset'] == 'ogbn-arxiv') or (epoch >= 50 and param['dataset'] != 'ogbn-arxiv' and param['teacher'] != 'GCN') or (epoch > 50 and param['dataset'] != 'ogbn-arxiv' and param['teacher'] == 'GCN'):
             if param['exp_setting'] == 'tran':
                 KD_model.updata(out_t_all.detach().cpu().numpy(), logits_s.detach().cpu().numpy())
             else:
                 KD_model.updata(obs_out_t.detach().cpu().numpy(), logits_s.detach().cpu().numpy())
             KD_prob = KD_model.predict_prob()
             sampling_mask = torch.ones(edge_idx_list[1].shape, dtype=torch.bool)
             edge_idx = torch.masked_select(edge_idx_list[1], sampling_mask).view(2, -1).detach().cpu().numpy().swapaxes(1, 0)
             edge_idx = edge_idx.tolist()
             for i in range(num_node):
                 edge_idx.append([i, i])
             edge_idx = np.array(edge_idx).swapaxes(1, 0)            
            

        if param['exp_setting'] == 'tran':
            loss_l, loss_t = train_mini_batch(model, edge_idx, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx_train, adj_mat_final_s, Delta,lap_if,node_energy_teacher, param)
            logits_s, out = evaluate_mini_batch(model, feats)
            IF_s = get_IF(logits_s,feats, adj_mat_final_s)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])
            

        else:
            loss_l, loss_t = train_mini_batch(model, edge_idx, obs_feats, obs_labels, obs_out_t, criterion_l, criterion_t, optimizer, obs_idx_train, param)
            logits_s, obs_out = evaluate_mini_batch(model, obs_feats)
            train_acc = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            val_acc = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
            _, out = evaluate_mini_batch(model, feats)
            test_acc = evaluator(out[idx_test_ind], labels[idx_test_ind])

          
        if epoch % 1 == 0:
            print("\033[0;30;43m [{}] CLA: {:.5f}, KD: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Power: {:.4f}  | IF: {:.4f}\033[0m".format(
                                        epoch, loss_l, loss_t, loss_l + loss_t, train_acc, val_acc, test_acc, val_best, test_val, test_best, KD_model.power,IF_s))
  

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            best_IF_s = IF_s  
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break

    model.load_state_dict(state)
    model.eval()
    print(f"Best Validation Accuracy: {val_best:.4f}, Test Accuracy: {test_val:.4f}, IF: {best_IF_s:.4f}")

    if param['exp_setting'] == 'tran':
        out, _ = evaluate_mini_batch(model, feats)
                
    else:
        obs_out, _ = evaluate_mini_batch(model, obs_feats)
        out, _ = evaluate_mini_batch(model, feats)
        out[idx_obs] = obs_out
    
    return out, test_acc, test_val, test_best, state,best_IF_s




