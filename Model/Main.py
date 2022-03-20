import tensorflow as tf
from helper.evaluate import *
from time import time
from JPPREC import JPPREC
import numpy as np
from tqdm import tqdm
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'


def makeDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    assert expected_order in ['acc', 'dec']
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

if __name__ == '__main__':
    np.set_printoptions(suppress=True, threshold=1000000)
    tf.set_random_seed(2021)
    np.random.seed(2021)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities
    config['A_in'] = sum(data_generator.lap_list) 
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['all_v_list'] = data_generator.all_v_list
    t0 = time()
    
    # Build model JPPRec
    model = JPPREC(data_config=config, args=args)
    saver = tf.train.Saver()

    # Save the model weights 
    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        makeDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    # Performance logger
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    max_friends = 0
    friends = {}
    beibei_flag = False
    if args.dataset == 'beibei':
        beibei_flag = True

    # load social relations
    with open("../Data/" + args.dataset + "/social_relation.txt", "r") as f:
        for line in tqdm(f):
            u1, u2 = [int(tmp) for tmp in line.strip().split("\t")]
            if u1 in friends:
                friends[u1].append(u2)
            else:
                friends[u1] = [u2]
            max_friends = max(max_friends, len(friends[u1]))
            if beibei_flag == True:
                if u2 in friends:
                    friends[u2].append(u1)
                else:
                    friends[u2] = [u1]
                max_friends = max(max_friends, len(friends[u2]))

    # load training data
    max_prtcs = 0
    prtcs = {}
    with open("../Data/" + args.dataset + "/train_id.txt", "r") as f:
        for line in tqdm(f):
            line = [int(tmp) for tmp in line.strip().split("\t")]
            prtcs[(line[0], line[1])] = line[2:]
            max_prtcs = max(max_prtcs, len(line[2:]))

    # load test data
    max_prtcs_test = 0
    prtcs_test = {}
    with open("../Data/" + args.dataset + "/test.txt", "r") as f:
        for line in tqdm(f):
            line = [int(tmp) for tmp in line.strip().split(" ")]
            prtcs_test[(line[0], line[1])] = line[2:]
            max_prtcs_test = max(max_prtcs_test, len(line[2:]))

    # Model training and testing
    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, item_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            btime= time()

            # prodrec batch
            batch_data = data_generator.generate_train_batch()
            batch_size = len(batch_data['users'])

            # partrec batch
            friends_batch = -np.ones((batch_size, max_friends)) 
            prtcs_batch = np.zeros((batch_size, max_friends))

            # load participants
            for i in range(batch_size):
                init = batch_data['users'][i]
                item = batch_data['pos_items'][i]
                if init not in friends:
                    friend = []
                else:
                    friend = friends[init]
                if (init, item) in prtcs:
                    prtc = prtcs[(init, item)]
                else:
                    prtc = []
                friends_batch[i, :len(friend)] = friend
                if len(friend) < 2:
                    continue
                for one_prtc in prtc:
                    prtcs_batch[i][friends_batch[i] == one_prtc] = 1

            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)
            feed_dict[model.friends] = friends_batch
            feed_dict[model.prtcs] = prtcs_batch
            friends_mask = np.zeros(friends_batch.shape)
            friends_mask[friends_batch == -1] = float("-inf")
            feed_dict[model.friends_mask] = friends_mask

            # model training 
            _, batch_loss, batch_base_loss, batch_item_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)
            tmp1, tmp2, tmp3, tmp4 = sess.run([model.tmp1, model.tmp2, model.tmp3, model.loss], feed_dict=feed_dict)
            loss += batch_loss
            base_loss += batch_base_loss
            item_loss += batch_item_loss
            reg_loss += batch_reg_loss  

        n_A_batch = len(data_generator.all_h_list) // args.batch_size_item + 1            
        model.update_attentive_A(sess)
        report_step = 10
        if (epoch + 1) % report_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, reg_loss)
                print(perf_str)
            continue

        # model testing 
        t2 = time()
        users_to_test = list(data_generator.test_user_dict.keys())
        ret = test(sess, model, users_to_test, friends, prtcs_test, max_friends, max_prtcs_test, drop_flag=False, batch_test_flag=batch_test_flag)
        t3 = time()
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        # report model perfromance
        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, base_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
            print('ProdRec Recall',ret['recall'])
            print('ProdRec NDCG',ret['ndcg'])
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][2], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)

        # early stopping
        if should_stop == True:
            break
        if ret['ndcg'][2] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)
    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    # save the trained model
    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    makeDir(save_path)
    f = open(save_path, 'a')
    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s, use_att=%s'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.adj_type, args.use_att, final_perf))
    f.close()



