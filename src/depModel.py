import os,sys
from decoder import *
import dynet as dynet

word_dict = {}
label_dict = {}
pos_dict = {}
action_dict = {}

with open("data/vocabs.word") as f:
    lines = f.readlines()
    for line in lines:
        word, num = line.strip().split()
        word_dict[word] = int(num)
with open("data/vocabs.pos") as f:
    lines = f.readlines()
    for line in lines:
        word, num = line.strip().split()
        pos_dict[word] = int(num)
with open("data/vocabs.labels") as f:
    lines = f.readlines()
    for line in lines:
        word, num = line.strip().split()
        label_dict[word] = int(num)
with open("data/vocabs.actions") as f:
    lines = f.readlines()
    for line in lines:
        word, num = line.strip().split()
        action_dict[word] = int(num)

def num_words():
    return len(word_dict.keys())
def num_labels():
    return len(label_dict.keys())
def num_tags():
    return len(pos_dict.keys())
def num_actions():
    return len(action_dict.keys())

model = dynet.Model()
transfer = dynet.rectify
updater = dynet.AdamTrainer(model)

word_embed_dim, pos_embed_dim, lab_embed_dim = 64, 32, 32

word_embedding = model.add_lookup_parameters((num_words(), word_embed_dim))
tag_embedding = model.add_lookup_parameters((num_tags(), pos_embed_dim))
label_embedding = model.add_lookup_parameters((num_labels(), lab_embed_dim))

input_dim = 20 * word_embed_dim + 20 * pos_embed_dim + 12 * lab_embed_dim
#hidden_dim, minibatch_size = 200, 1000
hidden_dim, minibatch_size = 400, 1000
# define the hidden layer.
hidden_layer = model.add_parameters((hidden_dim, input_dim))
# define the hidden layer bias term and initialize it as constant 0.2.
hidden_layer_bias = model.add_parameters(hidden_dim, init=dynet.ConstInitializer(0.2))
# define the hidden layer.
hidden_layer2 = model.add_parameters((hidden_dim, hidden_dim))
# define the hidden layer bias term and initialize it as constant 0.2.
hidden_layer_bias2 = model.add_parameters(hidden_dim, init=dynet.ConstInitializer(0.2))
# define the output weight.
output_layer = model.add_parameters((num_actions(), hidden_dim))
# define the bias vector and initialize it as zero.
output_bias = model.add_parameters(num_actions(), init=dynet.ConstInitializer(0))

model.populate("part2.model")

def forward(features):
    word_ids, pos_ids, dep_label_ids = features[:20], features[20:40], features[40:]

    # extract word embeddings and tag embeddings from features
    word_embeds = [word_embedding[wid] for wid in word_ids]
    pos_embeds = [tag_embedding[tid] for tid in pos_ids]
    label_embeds = [label_embedding[lid] for lid in dep_label_ids]


    # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
    embedding_layer = dynet.concatenate(word_embeds + pos_embeds + label_embeds)

    # calculating the hidden layer
    #  converts a parameter to a matrix expression in dynetnet (its a dynetnet-specific syntax).
    hidden = transfer(hidden_layer * embedding_layer + hidden_layer_bias)

    # calculating the hidden layer2
    #  converts a parameter to a matrix expression in dynetnet (its a dynetnet-specific syntax).
    hidden2 = transfer(hidden_layer2 * hidden + hidden_layer_bias2)

    # calculating the output layer
    output = output_layer * hidden2 + output_bias

    # return a list of outputs
    return output


def generate_features(fields):
    word, pos, dep_label = fields[:20], fields[20:40], fields[40:]
    for i in range(20):
        if word[i] in word_dict:
            word[i] = word_dict[word[i]]
        else:
            word[i] = word_dict["<unk>"]
    for i in range(20):
        if pos[i] in pos_dict:
            pos[i] = pos_dict[pos[i]]
        else:
            pos[i] = pos_dict["<null>"]
    for i in range(12):
        dep_label[i] = label_dict[dep_label[i]]

    return word + pos + dep_label

class DepModel:
    def __init__(self):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # if you prefer to have your own index for actions, change this.
        action_dict = {}
        with open("data/vocabs.actions") as f:
            lines = f.readlines()
            for line in lines:
                word, num = line.strip().split()
                action_dict[word] = int(num)

        self.actions = [x[0] for x in sorted(action_dict.items(), key=lambda x:x[1])]

        #self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']

        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        scores = []
        fields = generate_features(str_features)
        result = forward(fields)
        #scores.append(np.argmax(result.value()))

        # change this part of the code.
        #return [0]*len(actions)
        return result.value()

if __name__=='__main__':
    m = DepModel()
    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)
