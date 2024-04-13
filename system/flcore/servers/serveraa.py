
import time
import copy
import random
from flcore.clients.clientaa import clientAA
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch





class FedAa(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        
#         print(f'args.model: {args.model}')
        
        self.clip_model_object = args.model
        
        self.global_model = copy.deepcopy(self.clip_model_object.aa)    # aa model
        
        
        # print(f'args.global_model: {self.global_model}')
        
        self.few_shot = args.few_shot
        
        self.set_clients(clientAA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                
                

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
        
    # added ------------------------------------------------------
    def test_metrics_fs(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        # for c in self.clients[0]:
            
        c = self.clients[0]    
        
        ct, ns, auc = c.test_metrics()
        tot_correct.append(ct*1.0)
        tot_auc.append(auc*ns)
        num_samples.append(ns)
            
            
        print(f'ct, ns, auc: {ct}, {ns}, {auc}')

        # ids = [c.id for c in self.clients]
        
        ids = [c.id]
        
        print(f'ids: {ids}')

        return ids, num_samples, tot_correct, tot_auc
    
    
    def evaluate(self, acc=None, loss=None):
        
        if self.few_shot:
            stats = self.test_metrics_fs()
            
            print(f'stats[1]: {stats[1]}')
            print(f'stats[2]: {stats[2]}')
            print(f'stats[3]: {stats[3]}')

            test_acc = sum(stats[2])*1.0 / len(stats[2])
            test_auc = sum(stats[3])*1.0 / len(stats[3])

            accs = 0
            aucs = 0
            
            
        else:
            stats = self.test_metrics()
        
            print(f'stats[1]: {stats[1]}')
            print(f'stats[2]: {stats[2]}')
            print(f'stats[3]: {stats[3]}')

            test_acc = sum(stats[2])*1.0 / len(stats[2])
            test_auc = sum(stats[3])*1.0 / len(stats[3])

            accs = statistics.stdev(stats[2])
            aucs = statistics.stdev(stats[3])
        
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(accs))
        print("Std Test AUC: {:.4f}".format(aucs))
    
    def send_models(self):
        # Instead of sending the whole model, only send the Attention Adapter 
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            global_aa_params = self.global_model
            
            client.set_parameters(global_aa_params)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
    def receive_models(self):
        # Receive only the Attention Adapter from each client

        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.aa)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples  
            
        
        # print(f'self.uploaded_weights: {self.uploaded_weights}')
        # for element in self.uploaded_weights:
        #     print(type(element))
            
        # print(f'self.uploaded_models[0]: {self.uploaded_models[0]}')
        # print(f'self.uploaded_models[1]: {self.uploaded_models[1]}')
        
        # print(f'self.uploaded_models length: {len(self.uploaded_models)}')
#         for key, tensor in self.uploaded_models[0].items():
#             print(f"Shape of '{key}': {tensor.shape}")
            
#         for key, tensor in self.uploaded_models[1].items():
#             print(f"Shape of '{key}': {tensor.shape}")
            
        
    
    # ------------------------------------------------------------
    
    
    
    