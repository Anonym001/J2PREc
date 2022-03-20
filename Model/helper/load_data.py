import collections
import numpy as np
import random as rd

class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        trust_file = path + '/social_relation.txt'
        
        items_file = path + '/all_items.txt'
        ptc_file = path + '/train_ptc.txt'
        ini_file = path + '/train_ini.txt'

        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        self.ini_data, self.ini_user_dict = self._load_ratings(ini_file)

        self.ptc_data, self.ptc_user_dict = self._load_ratings(ptc_file)
        self.trust_data, self.trust_user_dict = self._load_trusts(trust_file)
  
        self.test_data, self.test_user_dict = self._load_test(test_file)
        self.exist_users = self.train_user_dict.keys()

        self._statistic_ratings()

        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.item_data, self.item_dict, self.relation_dict= self._load_items(items_file) 
        
        self.batch_size_item = self.n_triples // (self.n_train // self.batch_size)
        self._print_data_info()

    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict
    def _load_test(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], [inters[1]]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict
    def _load_trusts(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split('\t')]

            u_id, f_id = inters[0], inters[1]      
            inter_mat.append([u_id, f_id])
            if u_id in user_dict:
                user_dict[u_id].append(f_id)
            else:
                user_dict[u_id]=[f_id]
        return np.array(inter_mat),user_dict




    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    def _load_items(self, file_name):
        def _construct_item(item_np):
            item = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in item_np:
                item[head].append((tail, relation))
                rd[relation].append((head, tail))
            return item, rd

        item_np = np.loadtxt(file_name, dtype=np.int32)
        item_np = np.unique(item_np, axis=0)
        self.n_entities = max(max(item_np[:, 0]), max(item_np[:, 2])) + 1
        self.n_triples = len(item_np)
        item_dict, relation_dict = _construct_item(item_np)
        return item_np, item_dict, relation_dict

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1
            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)


        return split_uids, split_state
