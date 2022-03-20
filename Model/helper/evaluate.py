import helper.metrics as metrics
from helper.parser import parse_args
import multiprocessing
import heapq
import numpy as np
import random
from helper.loader_jpprec import JPPREC_loader

cores = multiprocessing.cpu_count() // 2
args = parse_args()
Ks = eval(args.Ks)

data_generator = JPPREC_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

tunefilename = args.data_path + args.dataset + '/test.txt'
negative_filename = args.data_path + args.dataset + '/test.negative.txt'
test_negative_dict = {}
test_list = []
Ks = [1, 3, 5, 10, 20]

# Load testing data
with open(tunefilename,'r') as file:
    for l in file.readlines():
        lines = l.strip('\n').split(' ')
        user = lines[0]
        item = lines[1]
        pair = (user,item)
        test_list.append(pair)
idx = 0

# Load negative items for testing
with open(negative_filename,'r') as file:
    for l in file.readlines():
        lines = l.strip('\n').split('\t')
        items = []
        for item in lines:
            items.append(int(item))
        pair = test_list[idx]
        user = int(pair[0])
        idx+=1
        if user in test_negative_dict:
            test_negative_dict[user] += items
        else:
            test_negative_dict[user] = items
            
            
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]
    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))
    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    rating = x[0]
    u = x[1]
    try:
        training_items = data_generator.train_user_dict[u]
    except Exception:
        training_items = []
    
    # groud-truth item for u
    user_pos_test = data_generator.test_user_dict[u]
    
    # negative items
    test_items = test_negative_dict[u] + user_pos_test

    # testing
    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, users_to_test, friends, prtcs, max_friends, max_prtcs, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)
    u_batch_size = BATCH_SIZE * 4
    u_batch_size_ptc = BATCH_SIZE // 100
    i_batch_size = BATCH_SIZE
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    n_user_batchs_ptc = n_test_users // u_batch_size_ptc + 1
    count = 0
    count_f = 0
    batch_result_recall_o = []
    batch_result_recall = []
    batch_result_ndcg_o = []
    batch_result_ndcg = []
    
    # mini-batch testing ProdRec
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]

        if batch_test_flag:
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
        else:
            item_batch = range(ITEM_NUM)
            feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                               user_batch=user_batch,
                                                               item_batch=item_batch,
                                                               drop_flag=drop_flag) 
            rate_batch = model.eval(sess, feed_dict=feed_dict)
            rate_batch = rate_batch.reshape((-1, len(item_batch)))
          
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

    # mini-batch testing PartRec  
    for u_batch_id in range(n_user_batchs_ptc):
        start = u_batch_id * u_batch_size_ptc
        end = (u_batch_id + 1) * u_batch_size_ptc
        user_batch = test_users[start: end]

        if batch_test_flag:
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
        else:
            item_batch = range(ITEM_NUM)
            feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                               user_batch=user_batch,
                                                               item_batch=item_batch,
                                                               drop_flag=drop_flag)
            inits = []
            items = []
            for init in user_batch:
                user_pos_test = data_generator.test_user_dict[init]
                for item in user_pos_test:
                    inits.append(init)
                    items.append(item)

            batch_size = len(inits)
            friends_batch = -np.ones((batch_size, max_friends))
            prtcs_batch = np.zeros((batch_size, max_friends))
            
            # Generate testing participants
            for i in range(batch_size):
                init = inits[i]
                item = items[i]
                if init not in friends:
                    continue
                friend = friends[init]
                if (init, item) in prtcs:
                    prtc = prtcs[(init, item)]
                else:
                    prtc = []
                friends_batch[i, :len(friend)] = friend
                for one_prtc in prtc:
                    prtcs_batch[i][friends_batch[i] == one_prtc] = 1

            feed_dict = {model.users:inits, model.pos_items:items,model.mess_dropout: [0, 0, 0],model.node_dropout: [0]}
            feed_dict[model.friends] = friends_batch
            feed_dict[model.prtcs] = prtcs_batch
            friends_mask = np.zeros(friends_batch.shape)
            friends_mask[friends_batch == -1] = float("-inf")
            feed_dict[model.friends_mask] = friends_mask
            prtc_scores_original, prtc_scores = sess.run([model.prtc_scores_original, model.prtc_scores], feed_dict)

            # PartRec scores
            sort_index = np.argsort(-prtc_scores, -1)
            prtcs_batch = prtcs_batch[np.arange(batch_size).reshape(-1, 1), sort_index]

            recall = []
            for k in Ks:
                sum_users = 0
                for i in prtcs_batch:
                    all_pos_num = i.sum()
                    value = metrics.recall_at_k(i, k,all_pos_num)
                    sum_users += value
                recall.append(sum_users)
            batch_result_recall.append(recall)

            dcg = []
            for k in Ks:
                sum_users = 0
                for i in prtcs_batch:
                    value = metrics.ndcg_at_k(i, k)
                    sum_users += value
                dcg.append(sum_users)         
            batch_result_ndcg.append(dcg)

        count_f += prtcs_batch.shape[0]

    # PartRec recall
    average_recall = [0,0,0,0,0]
    for batch_ks in batch_result_recall:
        for i in range(len(batch_ks)):
            average_recall[i] += batch_ks[i]
    average_recall = [i/float(count_f) for i in average_recall]
    print('PartRec Recall', average_recall)

    # PartRec ndcg
    average_ndcg = [0,0,0,0,0]
    for batch_ks in batch_result_ndcg:
        for i in range(len(batch_ks)):
            average_ndcg[i] += batch_ks[i]
    average_ndcg = [i/float(count_f) for i in average_ndcg]
    print('PartRec NDCG', average_ndcg)

    assert count == n_test_users
    pool.close()
    return result

