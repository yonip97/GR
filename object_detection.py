import random
import time
from termcolor import colored
import tqdm
from torchvision.models.detection.faster_rcnn import FasterRCNN,FastRCNNPredictor
from torchvision.models import resnet50
import torchvision
from os import listdir
from os.path import isfile, join
from metrics import metric_calculation
from torchvision import transforms
import torch
from torchvision.models import resnet50
from torch import nn
import numpy as np
from torch.optim import Adam
import batch_gen
import os
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from torch.utils.data import TensorDataset
from model import *
class Tool_Trainer():
    def __init__(self,epochs,lr,model,classes_num,train_data_loader,test_data_loader,batch_size,batch_generator,device = 'cuda'):
        self.epochs = epochs
        self.model = model
        self.optimizer = Adam(model.parameters(),lr=lr,betas=(0.9,0.99),eps=1e-6)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.batch_size = batch_size
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.classes_num = classes_num
        self.batch_generator = batch_generator
        self.ground_truth_left_tools = '/datashare/APAS/transcriptions_tools_left_new/'
        self.ground_truth_right_tools = '/datashare/APAS/transcriptions_tools_right_new/'

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch}")
            if epoch == 10:
                print("he")
            correct = 0
            total = 0
            epoch_loss = 0
            for batch in tqdm.tqdm(self.train_data_loader):
                for video in batch:
                    video_input,video_left_target,video_right_target = video
                    video_input, video_left_target, video_right_target = video_input.to(self.device)\
                        ,video_left_target.long().to(self.device),video_right_target.long().to(self.device)
                    video_input = video_input.permute((1,0)).requires_grad_().unsqueeze(0)
                    # data = TensorDataset(video_input,video_left_target,video_right_target)
                    # data_loader = DataLoader(data,batch_size=32)
                    # for mini_batch in data_loader:
                    #     input_mini_batch,left_mini_batch_target,right_mini_batch_target = mini_batch
                    #     input_mini_batch, left_mini_batch_target, right_mini_batch_target = input_mini_batch.to(self.device)\
                    #         , left_mini_batch_target.type(torch.LongTensor).to(self.device), right_mini_batch_target.type(torch.LongTensor).to(self.device)
                    #     mini_batch_left_predictions,mini_batch_right_predictions = self.model.forward(input_mini_batch)
                    left_predictions,right_predictions = self.model.forward(video_input)
                    left_loss = 0
                    left_length = min(video_input.size(2),video_left_target.size(0))
                    left_predictions = left_predictions[:,:,:,:left_length]
                    video_left_target = video_left_target[:left_length]
                    for p in left_predictions:
                        left_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.classes_num),
                                              video_left_target.view(-1))
                        left_loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16))
                    right_length = min(video_input.size(2),video_right_target.size(0))
                    right_predictions = right_predictions[:,:,:,:right_length]
                    video_right_target = video_right_target[:right_length]
                    right_loss = 0
                    for p in right_predictions:
                        right_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.classes_num),
                                              video_right_target.view(-1))
                        right_loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16))
                    loss = left_loss+right_loss
                    loss = loss * 1/self.batch_size
                    loss.backward()
                    _, predicted_left = torch.max(left_predictions[-1].data, 1)
                    correct += (predicted_left == video_left_target).float().sum().item()
                    total += left_length
                    _, predicted_right = torch.max(right_predictions[-1].data, 1)
                    correct += (predicted_right == video_right_target).float().sum().item()
                    total += right_length
                    epoch_loss += loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            print(colored("Results", 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          train_data_loader),
                                                                                                      float(
                                                                                                      correct) / total))
            print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
            self.evaluate()

    def evaluate(self):
        results = {}
        #device = eval_dict["device"]
        #features_path = eval_dict["features_path"]
        #sample_rate = eval_dict["sample_rate"]
        #actions_dict_gesures = eval_dict["actions_dict_gestures"]
        #ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            list_of_vids = self.batch_generator.list_of_valid_examples
            recognition_left_list = []
            recognition_right_list = []
            for seq in list_of_vids:
                # print vid
                input_x = torch.load(features_path + seq + '_side.pt')
                #features = features[:, ::sample_rate]
                #input_x = torch.tensor(features, dtype=torch.float)
                input_x = input_x.to(self.device)
                #video_features,_ = batch_gen.get_video_data(seq.split('.')[0])
                #video_features = video_features.permute((1,0)).unsqueeze(0).to(self.device)
                input_x = input_x.permute((1,0)).unsqueeze(0)
                predictions_left,predictions_right = self.model.forward(input_x)
                #predictions1 = self.model(input_x, torch.tensor([features.shape[1]]))
                _,predictions_right = torch.max(torch.nn.Softmax(dim=1)(predictions_right[-1]),1)
                _,predictions_left = torch.max(torch.nn.Softmax(dim=1)(predictions_left[-1]),1)
                predictions_right = predictions_right.squeeze()
                predictions_left = predictions_left.squeeze()
                # predictions1 = predictions1[0].unsqueeze_(0)
                # predictions1 = torch.nn.Softmax(dim=2)(predictions1)




                recognition_right = []
                for i in range(len(predictions_right)):
                    recognition_right = np.concatenate((recognition_right, [list(self.batch_generator.action_dict_tools.keys())[
                                                                      list(self.batch_generator.action_dict_tools.values()).index(
                                                                          predictions_right[i].item())]] * 6))
                recognition_right_list.append(recognition_right)
                recognition_left = []
                for i in range(len(predictions_right)):
                    recognition_left = np.concatenate(
                        (recognition_left, [list(self.batch_generator.action_dict_tools.keys())[
                                                 list(self.batch_generator.action_dict_tools.values()).index(
                                                     predictions_left[i].item())]] * 6))
                recognition_left_list.append(recognition_left)


            print("gestures results")
            results_left, _ = metric_calculation(ground_truth_path=self.ground_truth_left_tools,
                                             recognition_list=recognition_left_list, list_of_videos=list_of_vids,
                                             suffix="tools")
            results_right,_ = metric_calculation(ground_truth_path=self.ground_truth_right_tools,
                                             recognition_list=recognition_right_list, list_of_videos=list_of_vids,
                                             suffix="tools")
            #results.update(results1)


            self.model.train()
            return results

class hand_model(nn.Module):
    def __init__(self,classes_num):
        super(hand_model,self).__init__()
        self.resnet_feature_extractor = resnet50(pretrained=True)
        self.embeddings_to_classes = nn.Linear(1000,classes_num)
    def forward(self,input):
        features_resnet = self.resnet_feature_extractor.forward(input)
        output = self.embeddings_to_classes.forward(features_resnet)
        return output

class tool_model(nn.Module):
    def __init__(self,num_stages,num_layers,num_f_maps,features_dim,classes_num,device='cuda'):
        super(tool_model,self).__init__()
        self.left_model = MultiStageModel(num_stages,num_layers,num_f_maps,features_dim,classes_num)
        self.right_model = MultiStageModel(num_stages,num_layers,num_f_maps,features_dim,classes_num)
        self.device = device

    def forward(self,input):
        left_output = self.left_model.forward(input)
        right_output = self.right_model.forward(input)
        return left_output,right_output



class data():
    def __init__(self,folds_folder,split_num,features_path,tools_path_right,tools_path_left,sample_rate,mapping_tool_file):
        self.folds_folder = folds_folder
        self.split_num = split_num
        self.features_path = features_path
        self.tools_path_right = tools_path_right
        self.tools_path_left = tools_path_left
        self.sample_rate = sample_rate
        self.mapping_tool_file = mapping_tool_file
        self.action_dict_tools = dict()
        self.classes_num = self.get_action_dict_tools()
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),normalizer])
        self.index = 0
        self.read_data()



    def get_action_dict_tools(self):
        file_ptr = open(self.mapping_tool_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            self.action_dict_tools[a.split()[1]] = int(a.split()[0])
        num_classes_tools = len(self.action_dict_tools)
        return num_classes_tools

    def read_data(self):
        self.list_of_train_examples =[]
        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "fold" in filename:
                if str(self.split_num) in filename:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.list_of_valid_examples = [f.split('.')[0] for f in file_ptr.read().split('\n')[:-1]]
                    file_ptr.close()
                    random.shuffle(self.list_of_valid_examples)
                else:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    temp = [f.split('.')[0] for f in file_ptr.read().split('\n')[:-1]]
                    self.list_of_train_examples = self.list_of_train_examples + temp
                    file_ptr.close()
                continue
            else:
                continue

    def pars_ground_truth(self, gt_source):
        contant = []
        for line in gt_source:
            info = line.split()
            line_contant = [info[2]] * (int(info[1]) - int(info[0]) + 1)
            contant = contant + line_contant
        return contant


    def create_data(self,list_of_files):
        target_left = []
        target_right = []
        input =[]
        counter = 0
        for seq in list_of_files:
            counter+=1
            images = torch.load(self.features_path + seq + '_side.pt').numpy()
            normalized_images = [self.transform(image.transpose((1,2,0))) for image in images]
            features = torch.stack(normalized_images)
            input.append(features)
            file_ptr_right = open(self.tools_path_right + seq + '.txt', 'r')
            gt_source_right = file_ptr_right.read().split('\n')[:-1]
            content_right = self.pars_ground_truth(gt_source_right)
            classes_size_right = min(features.size(0)*self.sample_rate, len(content_right))
            classes_right = torch.zeros(classes_size_right)
            for i in range(len(classes_right)):
                classes_right[i] = self.action_dict_tools[content_right[i]]

            target_right.append(classes_right[::self.sample_rate])

            file_ptr_left = open(self.tools_path_left +seq + '.txt', 'r')
            gt_source_left = file_ptr_left.read().split('\n')[:-1]
            content_left = self.pars_ground_truth(gt_source_left)
            classes_size_left = min(features.size(0)*self.sample_rate, len(content_left))
            classes_left = torch.zeros(classes_size_left)
            for i in range(len(classes_left)):
                classes_left[i] = self.action_dict_tools[content_left[i]]

            target_left.append(classes_left[::self.sample_rate])
# input = torch.stack(input)
        # target_left = torch.stack(target_left)
        # target_right = torch.stack(target_right)
        # data = torch.utils.data.TensorDataset(input,target_left,target_right)
        # data_loader = DataLoader(data,batch_size=batch_size,shuffle=True)
        return [input,target_left,target_right]


    def create_data_loader(self,list_of_files,batch_size):
        target_left = []
        target_right = []
        input = []
        input_lengths = []
        for seq in list_of_files:
            features = torch.load(self.features_path + seq + '_side.pt')
            input.append(features)
            input_lengths.append(features.size(0))
            file_ptr_right = open(self.tools_path_right + seq + '.txt', 'r')
            gt_source_right = file_ptr_right.read().split('\n')[:-1]
            content_right = self.pars_ground_truth(gt_source_right)
            classes_size_right = min(features.size(0) * self.sample_rate, len(content_right))
            classes_right = torch.zeros(classes_size_right)
            for i in range(len(classes_right)):
                classes_right[i] = self.action_dict_tools[content_right[i]]

            target_right.append(classes_right[::self.sample_rate])

            file_ptr_left = open(self.tools_path_left + seq + '.txt', 'r')
            gt_source_left = file_ptr_left.read().split('\n')[:-1]
            content_left = self.pars_ground_truth(gt_source_left)
            classes_size_left = min(features.size(0) * self.sample_rate, len(content_left))
            classes_left = torch.zeros(classes_size_left)
            for i in range(len(classes_left)):
                classes_left[i] = self.action_dict_tools[content_left[i]]

            target_left.append(classes_left[::self.sample_rate])
        input = torch.stack(input)
        target_left = torch.stack(target_left)
        target_right = torch.stack(target_right)
        data = torch.utils.data.TensorDataset(input,target_left,target_right)
        data_loader = DataLoader(data,batch_size=batch_size,shuffle=True)
        return data_loader
class Custom_data_set(torch.utils.data.Dataset):
    def __init__(self,files,sample_rate,mapping_tool_file,features_path,tools_path_left,tools_path_right):
        super(Custom_data_set, self).__init__()
        self.files = files
        self.features_path = features_path
        self.tools_path_right =tools_path_right
        self.tools_path_left = tools_path_left
        self.sample_rate = sample_rate
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),normalizer])
        self.mapping_tool_file = mapping_tool_file
        self.action_dict_tools = {}
        self.num_classes = self.get_action_dict_tools()

    def __getitem__(self, index):
        file = self.files[index]
        return self.load(file)

    def __len__(self):
        return len(self.files)

    def get_action_dict_tools(self):
        file_ptr = open(self.mapping_tool_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            self.action_dict_tools[a.split()[1]] = int(a.split()[0])
        num_classes_tools = len(self.action_dict_tools)
        return num_classes_tools

    def pars_ground_truth(self, gt_source):
        contant = []
        for line in gt_source:
            info = line.split()
            line_contant = [info[2]] * (int(info[1]) - int(info[0]) + 1)
            contant = contant + line_contant
        return contant

    def load(self,file):
        features = torch.load(self.features_path + file + '_side.pt')
        file_ptr_right = open(self.tools_path_right + file + '.txt', 'r')
        gt_source_right = file_ptr_right.read().split('\n')[:-1]
        content_right = self.pars_ground_truth(gt_source_right)
        classes_size_right = min(features.size(0) * self.sample_rate, len(content_right))
        classes_right = torch.zeros(classes_size_right)
        for i in range(len(classes_right)):
            classes_right[i] = self.action_dict_tools[content_right[i]]
        target_right = classes_right[::self.sample_rate]

        file_ptr_left = open(self.tools_path_left + file + '.txt', 'r')
        gt_source_left = file_ptr_left.read().split('\n')[:-1]
        content_left = self.pars_ground_truth(gt_source_left)
        classes_size_left = min(features.size(0) * self.sample_rate, len(content_left))
        classes_left = torch.zeros(classes_size_left)
        for i in range(len(classes_left)):
            classes_left[i] = self.action_dict_tools[content_left[i]]

        target_left = classes_left[::self.sample_rate]
        return features,target_left,target_right
def collate_fn(input):
    return input


folds_folder = "/datashare/APAS/folds"
features_path ='/home/student/Desktop/tensors/'
path_tools_left = "/datashare/APAS/transcriptions_tools_left/"
path_tools_right = "/datashare/APAS/transcriptions_tools_right/"
mapping_tools_file = "/datashare/APAS/mapping_tools.txt"

start = time.time()


# start = time.time()
# x,y,z = batch_generator.next_batch_with_gt_tools_as_input(70)
# print(time.time()-start)
epochs = 40
lr = 0.0005
classes_num= 4
batch_size = 1
for split in [0,1,2,3,4]:
    batch_generator = data(folds_folder, split, features_path, path_tools_right, path_tools_left, 6, mapping_tools_file)
    train_data_set = Custom_data_set(batch_generator.list_of_train_examples, 6, mapping_tools_file, features_path,
                                     path_tools_left, path_tools_right)
    #train_data_loader = batch_generator.create_data_loader(batch_generator.list_of_train_examples,5)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_data_set = Custom_data_set(batch_generator.list_of_valid_examples,6,mapping_tools_file,features_path,path_tools_left,path_tools_right)
    test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)
    model =tool_model(4,10,65,1000,classes_num)
    trainer = Tool_Trainer(epochs,lr,model,classes_num,train_data_loader,test_data_loader,batch_size,batch_generator)
    trainer.train()

# mapping_tool_file = "/datashare/APAS/mapping_tools.txt"
# actions_dict_tools = dict()
# file_ptr = open(mapping_tool_file, 'r')
# actions = file_ptr.read().split('\n')[:-1]
# file_ptr.close()
# for a in actions:
#     actions_dict_tools[a.split()[1]] = int(a.split()[0])
# num_classes_tools = len(actions_dict_tools)
# path = '/home/student/Desktop/images_tensors/'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# files = [f for f in listdir(path) if isfile(join(path, f))]
# transform = transforms.Compose([transforms.ToTensor(),normalize])
# model_left_hand =resnet50(pretrained=True)
# embeddings_to_classes_left_hand = nn.Linear(1000,6)
# model_right_hand = resnet50(pretrained=True)
# embeddings_to_classes_right_hand = nn.Linear(1000,6)
# left_hand_path =  '/datashare/APAS/transcriptions_tools_left_new/'
# right_hand_path ='/datashare/APAS/transcriptions_tools_right_new/'
# ce = nn.CrossEntropyLoss()
# optimizer_right_hand = Adam(model_right_hand.parameters(),betas = (0.9,0.99),lr = 0.0005)
# optimizer_left_hand = Adam(model_left_hand.parameters(),betas = (0.9,0.99),lr = 0.0005)
# for file in files:
#     images = torch.load(path+file).numpy()
#
#     file_ptr_left_hand = open(left_hand_path +file.split('.')[0]+ '.txt', 'r')
#     gt_source = file_ptr_left_hand.read().split('\n')[:-1]
#     content = pars_ground_truth(gt_source)
#     classes_size = min(np.shape(images)[1], len(content))
#
#     classes = np.zeros(classes_size)
#     for i in range(len(classes)):
#         classes[i] = actions_dict_tools[content[i]]
#     left_hand_target = classes[::6]
#
#     file_ptr_left_hand = open(right_hand_path +file.split('.')[0]+ '.txt', 'r')
#     gt_source = file_ptr_left_hand.read().split('\n')[:-1]
#     content = pars_ground_truth(gt_source)
#     classes_size = min(np.shape(images)[1], len(content))
#
#     classes = np.zeros(classes_size)
#     for i in range(len(classes)):
#         classes[i] = actions_dict_tools[content[i]]
#     right_hand_target = classes[::6]
#
#     normalized_images = [transform(image) for image in images]
#     normalized_images = torch.stack(normalized_images)
#     output_left_hand = embeddings_to_classes_left_hand(model_left_hand.forward(normalized_images))
#     output_right_hand = embeddings_to_classes_right_hand(model_right_hand.forward(normalized_images))
#     left_loss = ce.forward(output_left_hand,left_hand_target)
#     right_loss = ce.forward(output_right_hand,right_hand_target)
#     loss = left_loss+right_loss
#     loss.backwards()
#     optimizer_right_hand.step()
#     optimizer_left_hand.step()

# backbone = resnet50(pretrained=True)
# backbone.out_channels = 1000
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))
# FasterRCNN(resnet50(pretrained=True),)
