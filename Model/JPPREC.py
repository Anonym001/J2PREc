import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class JPPREC(object):

    def __init__(self, data_config, args):
        self._parse_args(data_config, args)
        self.node_dropout_flag=1
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model()
        self._build_loss()
        self._attention_weight()
        self._update()
        self._statistics_params()

    def _parse_args(self, data_config,  args):
       
        self.model_type = 'jpprec'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']
        self.n_fold = 100
        self.A_in = data_config['A_in'] 
        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']
       
        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.batch_size_item= args.batch_size_item

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose

    def _build_inputs(self):
        
        # build inputs for product recommendation
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
  
        # build inputs for participant recommendation
        self.friends = tf.placeholder(tf.int32, shape=(None, None))
        self.prtcs = tf.placeholder(tf.int32, shape=(None, None))
        self.friends_mask = tf.placeholder(tf.float32, shape=(None, None))

        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):

        model_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        self.weight_size_list = [self.emb_dim] + self.weight_size
        self.n_output = self.n_layers+1

        # user and item embeddings
        model_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
        model_weights['item_embed'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embed')
 
        # relation embeddings
        model_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.emb_dim]),name='relation_embed')
        model_weights['relation_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim*2, self.emb_dim]))

        # bulid weights for multi-head attention
        self.n_heads =3
        self.head_dim = 64
        model_weights['query_ui1'] = tf.Variable(initializer([self.emb_dim * self.n_output*2, self.emb_dim*self.n_output]))
        model_weights['query_ui2'] = tf.Variable(initializer([self.emb_dim * self.n_output, self.n_heads*self.head_dim]))
        model_weights['mul_key_p1'] = tf.Variable(initializer([self.emb_dim * self.n_output, self.n_heads*self.head_dim]))
        
        # build weights for each propagation layer
        for k in range(self.n_layers):

            # gated influence aggregation
            model_weights['gating_W1_%d' % k] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
            model_weights['gating_W2_%d' % k] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
            
            # node-specific transformation
            model_weights['user_W_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_%d' % k)
            model_weights['user_b_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_%d' % k)
            model_weights['item_W_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_%d' % k)
            model_weights['item_b_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_%d' % k)

        return model_weights

    def _build_model(self):

        self.ua_embeddings, self.ea_embeddings = self._create_jpprec_interaction_embed()
        self.u_e = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.pos_items)
        self.neg_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.neg_items)
        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)

    def _attention_weight(self):    
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_item_inference(self.h, self.r, self.pos_t, self.neg_t)
        self.A_att_score = self._generate_attention_score(h=self.h, t=self.pos_t, r=self.r)
        self.A_out = self._create_attentive_A_out()

    def _get_item_inference(self, h, r, pos_t, neg_t):

        embeddings = tf.concat([self.weights['user_embed'], self.weights['item_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)
        h_e = tf.nn.embedding_lookup(embeddings, h)
        pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)
        relation_M = tf.nn.embedding_lookup(self.weights['relation_W'], r)
        return h_e, r_e, pos_t_e, neg_t_e

    def _build_loss(self):

        # calculate recommendation loss
        pos_scores = tf.reduce_sum(tf.multiply(self.u_e, self.pos_i_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_e, self.neg_i_e), axis=1)
        base_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        self.base_loss = base_loss

        # calculate participant recommendation loss
        ua_embeddings = tf.concat([self.ua_embeddings, tf.zeros([1, self.emb_dim * self.n_output])], 0)
        self.friend_e = tf.nn.embedding_lookup(ua_embeddings, self.friends)
  
        self.prtc_scores_original = tf.reduce_mean(self.friend_e * tf.expand_dims(self.pos_i_e, 1), -1)
        self.prtc_scores_original += self.friends_mask
        self.tmp1 = tf.nn.softmax(self.prtc_scores_original, -1)
        
        # define query vector
        u_conc_i = tf.concat([self.u_e,self.pos_i_e],-1)
        u_conc_i = tf.nn.relu(tf.matmul(u_conc_i, self.weights["query_ui1"])) 
        self.query =  u_conc_i
        self.query = tf.expand_dims(self.query,1)
        self.query = tf.matmul(self.query, self.weights["query_ui2"])
   
        # define key vectors
        self.key = tf.matmul(self.friend_e, self.weights["mul_key_p1"]) 

        # multi-head attention mechanism
        self.query_mul = tf.concat(tf.split(self.query, self.n_heads, axis=2), axis=-1)
        self.key_mul = tf.concat(tf.split(self.key, self.n_heads, axis=2), axis=-1)
        self.prtc_scores = tf.reduce_mean(self.query_mul * self.key_mul, -1)
        self.prtc_scores += self.friends_mask
        self.tmp2 = tf.nn.softmax(self.prtc_scores, -1)
        self.prtc_scores = tf.nn.log_softmax(self.prtc_scores, -1)
     
        self.tmp3 = self.prtc_scores[tf.cast(self.prtcs, tf.bool)]
        self.prtc_loss = -tf.reduce_mean(self.prtc_scores[tf.cast(self.prtcs, tf.bool)])
        
        # L2 regularization
        regularizer = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)
        regularizer = regularizer / self.batch_size
        self.reg_loss = self.regs[0] * regularizer
        self.item_loss = tf.constant(0.0, tf.float32, [1])

        # Overall loss function     
        self.loss = self.base_loss + 0.0001* self.prtc_loss + self.reg_loss 

        # Optimization with Adam
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        
    # relational graph network
    def _create_jpprec_interaction_embed(self):

        A = self.A_in

        # drop-out strategy
        if self.node_dropout_flag:
            A_fold = self._split_A_node_dropout(A)
        else:
            A_fold = self._split_A(A)
        
        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['item_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            ua_embeddings, ea_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_entities], 0)

            # node-specific transformation
            ua_embeddings = tf.matmul(ua_embeddings, self.weights['user_W_%d' % k]) + self.weights['user_b_%d' % k]
            ea_embeddings = tf.matmul(ea_embeddings, self.weights['item_W_%d' % k]) + self.weights['item_b_%d' % k]
            ego_embeddings = tf.concat([ua_embeddings, ea_embeddings], axis=0)

            # relation-aware influence propogation
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold[f], ego_embeddings))
       
            side_embeddings = tf.concat(temp_embed, 0)
           
            # gated influence aggregation 
            gating = tf.nn.sigmoid(tf.matmul(ego_embeddings, self.weights['gating_W1_%d' % k]) + tf.matmul(side_embeddings, self.weights['gating_W2_%d' % k]))     
            bi_embeddings = tf.multiply(gating, side_embeddings)
            inverse_gating = tf.ones([1,self.emb_dim],tf.float32)-gating
            pre_embeddings = tf.multiply(inverse_gating, ego_embeddings)
            ego_embeddings = bi_embeddings + pre_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            # layer concatenation operation 
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return ua_embeddings, ea_embeddings


    # Generating relation-aware influence weight 
    def _generate_attention_score(self, h, t, r):

        embeddings = tf.concat([self.weights['user_embed'], self.weights['item_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)
        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)
        
        # relation embedding and trainable weights
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)
        relation_M = tf.nn.embedding_lookup(self.weights['relation_W'], r) 

        # relation-aware attention mechanism       
        concate = tf.concat([h_e, t_e], axis=-1)
        con = tf.nn.relu(tf.reshape(tf.matmul(concate, relation_M),[-1, self.emb_dim]))
        attention_score = tf.reduce_sum(tf.multiply(con, r_e),1)

        return attention_score


    # Defining other helper functions
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _statistics_params(self):
        
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.item_loss, self.reg_loss], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A

    def _update(self):
        def _get_item_score(h_e, r_e, t_e):
            item_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return item_score

        pos_item_score = _get_item_score(self.h_e, self.r_e, self.pos_t_e)
        neg_item_score = _get_item_score(self.h_e, self.r_e, self.neg_t_e)
        
        item_loss = tf.reduce_mean(tf.nn.softplus(-(neg_item_score - pos_item_score)))

        item_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        item_reg_loss = item_reg_loss / self.batch_size_item
        self.item_loss2 = item_loss
        self.reg_loss2 = self.regs[1] * item_reg_loss
        self.loss2 = self.item_loss2 + self.reg_loss2
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):

        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)      
    
    # We refer to the implementation in KGAT for drop-out strategy
    def _split_A_node_dropout(self, X):
        A_fold = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold

    def _split_A(self, X):
        A_fold = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold

    def train_A(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.item_loss2, self.reg_loss2], feed_dict)

    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        att_score = []
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            A_att_score = sess.run(self.A_att_score, feed_dict=feed_dict)
            att_score += list(A_att_score)

        att_score = np.array(att_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: att_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_items,
                                                                       self.n_users + self.n_items))


