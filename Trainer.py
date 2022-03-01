#Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import time

import torch

from model import *
import sys
from torch import optim
import math
import pandas as pd
from termcolor import colored, cprint

from metrics import*
import wandb
from datetime import datetime
import tqdm
from torchvision.models import resnet50
from torch.utils.data import DataLoader

class Trainer:
    # def __init__(self, dim, num_classes_list,hidden_dim=64,dropout=0.4,num_layers=3, offline_mode=True, task="gestures", device="cuda",
    #              network='LSTM',debagging=False):
    def __init__(self,mstcn_num_stages,mstcn_num_layers,mstcn_num_f_maps,images_features_dim,num_classes,rnn_input_dim,
                   rnn_hidden_dim = 64,dropout =0.4,rnn_layers = 3,offline_mode = True,lamb = 1,device = "cuda",network="GRU",debagging = False):
        # self.model = MT_RNN_dp(network, input_dim=dim, hidden_dim=hidden_dim, num_classes_list=num_classes_list,
        #                     bidirectional=offline_mode, dropout=dropout,num_layers=num_layers)
        self.model = hybrid_model(mstcn_num_stages,mstcn_num_layers,mstcn_num_f_maps,images_features_dim,num_classes
                                  ,network,rnn_input_dim,rnn_hidden_dim,offline_mode,dropout,rnn_layers,device)
        #self.model = hybrid_model(4,10,65,1000,6,network,rnn_input_dim,rnn_hidden_dim,offline_mode,dropout,rnn_layers)
        #self.video_model = MultiStageModel(4,10,65,1000,6)
        #self.video_model = MS_TCN2(11,10,3,64,1000,6)
        self.debagging = debagging
        self.network = network
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.lamb = lamb


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):

        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.init(group= args.group ,
                       name="split: " + args.split,entity="yuvalyoni",
                       reinit=True)
            delattr(args, 'split')
            wandb.config.update(args)

        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0

            while batch_gen.has_next():
                batch_input, batch_target_rnn, mask,batch_videos_input,batch_target_mstcn = batch_gen.next_batch(batch_size)
                batch_input, batch_target_rnn, mask = batch_input.to(self.device), batch_target_rnn.to(
                    self.device), mask.to(self.device)
                batch_videos_input = [video.permute((1,0)).requires_grad_().unsqueeze(0).to(self.device) for video in batch_videos_input]
                batch_target_mstcn = [target.to(self.device) for target in batch_target_mstcn]
                optimizer.zero_grad()
                lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                mctsn_outputs,final_predictions = self.model.forward(batch_videos_input,batch_input,lengths)
                #predictions1 = self.model(batch_input, lengths)
                predictions1 = (final_predictions[0] * mask).unsqueeze_(0)

                mstcn_loss = 0
                rnn_loss = 0
                for p in predictions1:
                    rnn_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                    batch_target_rnn.view(-1))
                for video,target in zip(mctsn_outputs,batch_target_mstcn):
                    for p in video:
                        mstcn_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), target.view(-1))
                        mstcn_loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16))
                loss = rnn_loss + self.lamb * mstcn_loss
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted1 = torch.max(predictions1[-1].data, 1)
                for i in range(len(lengths)):
                    correct1 += (predicted1[i][:lengths[i]] == batch_target_rnn[i][
                                                               :lengths[i]]).float().sum().item()
                    total1 += lengths[i]

                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            if not self.debagging:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(colored(dt_string, 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          batch_gen.list_of_train_examples),
                                                                                                      float(
                                                                                                          correct1) / total1))
            train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                             "train acc": float(correct1) / total1}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload is True:
                    wandb.log(results)

        return eval_results_list, train_results_list

    def train_videos(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):

        number_of_batches= len(batch_gen.list_of_train_video_examples)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.init(project=args.project, group= 'MS-TCN2_top ' + args.group,
                       name="split: " + args.split,
                       reinit=True)
            delattr(args, 'split')
            wandb.config.update(args)

        self.video_model.train()
        self.video_model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.video_model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                optimizer.zero_grad()
                batch_input_features, batch_target_gestures = batch_gen.next_batch_video()
                batch_input_features,batch_target_gestures = batch_input_features.to(self.device),batch_target_gestures.to(self.device)
                batch_input_features = batch_input_features.permute((1,0)).requires_grad_().unsqueeze(0).to(self.device)

                predictions = self.video_model(batch_input_features)
                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target_gestures.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16))
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                correct += (predicted == batch_target_gestures).sum()
                total += len(predicted)
                pbar.update(1)
            batch_gen.reset()
            pbar.close()
            if not self.debagging:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(colored(dt_string, 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          batch_gen.list_of_train_examples),
                                                                                                      float(
                                                                                                          correct) / total))
            train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                             "train acc": float(correct) / total}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                results.update(self.video_eval(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload is True:
                    wandb.log(results)

        return eval_results_list, train_results_list


    def evaluate(self, eval_dict, batch_gen):
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        sample_rate = eval_dict["sample_rate"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            list_of_vids = batch_gen.list_of_valid_examples
            recognition1_list = []

            for seq in list_of_vids:
                # print vid
                features = np.load(features_path + seq.split('.')[0] + '.npy')
                if batch_gen.normalization is not None:
                    features = batch_gen.normalize(features)
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                video_features,_ = batch_gen.get_video_data(seq.split('.')[0])
                video_features = video_features.permute((1,0)).unsqueeze(0).to(self.device)
                _,predictions1 = self.model.forward([video_features],input_x,[torch.tensor([features.shape[1]])])
                #predictions1 = self.model(input_x, torch.tensor([features.shape[1]]))
                predictions1 = predictions1[0].unsqueeze_(0)
                predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                _, predicted1 = torch.max(predictions1[-1].data, 1)
                predicted1 = predicted1.squeeze()


                recognition1 = []
                for i in range(len(predicted1)):
                    recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                                                                      list(actions_dict_gesures.values()).index(
                                                                          predicted1[i].item())]] * sample_rate))
                recognition1_list.append(recognition1)

            print("gestures results")
            results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                             recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                             suffix="gesture")
            results.update(results1)


            self.model.train()
            return results

    def video_eval(self,eval_dict,batch_gen):
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.video_model.eval()
        with torch.no_grad():
            self.video_model.to(self.device)
            list_of_vids = batch_gen.list_of_valid_video_examples
            recognition_list = []
            for video_features_tensor in batch_gen.get_eval_videos():
                input_features = video_features_tensor.permute((1, 0))
                # print vid
                input_features = input_features.to(self.device).unsqueeze(0)
                predictions = self.video_model(input_features)

                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()


                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict_gesures.keys())[
                                                                      list(actions_dict_gesures.values()).index(
                                                                          predicted[i].item())]] * 6))
                recognition_list.append(recognition)

            print("gestures results")
            results, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                             recognition_list=recognition_list, list_of_videos=list_of_vids,
                                             suffix="gesture")
            results.update(results)


            self.video_model.train()
            return results


