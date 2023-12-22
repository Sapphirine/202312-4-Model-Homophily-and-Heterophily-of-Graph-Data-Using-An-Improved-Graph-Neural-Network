import torch
from torch_geometric.data import Data
from data_loader import data_loaders
from hyperparameters_setting import *
from utils.statistic import *

def generate_dataset(data_dict: dict):
    # Extracting node features
    node_features = [node["features"] for node in data_dict["nodes"]]
    x = torch.tensor(node_features, dtype=torch.float)

    # Extracting edge information
    edges = [(edge["source"], edge["target"]) for edge in data_dict["edges"]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data




def perform_inference(dataset): 
    args = HyperparameterSetting("node classification").get_args()
    args.dataset = generate_dataset(dataset)
    args.dataset['num_classes'] = 7
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MODEL_CLASSES["GraphSage"](args).to(args.device)
    model.load_state_dict(torch.load('/home/gemo/workspace/Graph-Neural-Network/saved/model/Cora/gbk/[0.6, 0.2, 0.2]/GraphSage/save.pt'))
    model.eval()
    pred_list = []

    for i in range(len(args.dataset)):
        data = args.dataset[i].to(args.device)
        _, pred = model(data)[0].max(dim=1)

        pred_list += pred.tolist()
    return pred_list


def perform_inference_old(): 
    args = HyperparameterSetting("node classification").get_args()
    args.dataset = data_loaders.DataLoader(args).dataset
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = args.dataset["graph"][0].edge_index
    args.similarity = compute_cosine_similarity(args.dataset, edge_index, "label")
    
    model = MODEL_CLASSES["GraphSage"](args).to(args.device)
    model.load_state_dict(torch.load('/home/gemo/workspace/Graph-Neural-Network/saved/model/Cora/gbk/[0.6, 0.2, 0.2]/GraphSage/save.pt'))
    model.eval()
    pred_list = []

    for i in range(len(args.dataset['graph'])):
        data = args.dataset['graph'][i].to(args.device)
        _, pred = model(data)[0].max(dim=1)

        pred_list += pred.tolist()
    return pred_list

def inference(args, model): 
    model.eval()
    pred_list = []

    for i in range(len(args.dataset['graph'])):
        data = args.dataset['graph'][i].to(args.device)
        _, pred = model(data)[0].max(dim=1)

        pred_list += pred.tolist()

    return pred_list

def main():
    pred = perform_inference()
    print(len(pred), pred)

if __name__ == '__main__':
    main()