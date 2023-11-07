import numpy as np
import pickle


# rating_file = '../data/' + 'movie' + '/ratings_final'
# rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
# a=[]
# for item in rating_np:
#     if item[2] == 1:
#         a.append(item)
# b=np.array(a)
# np.save('../data/' + 'last-fm-small' + '/rating_pos' + '.npy', rating_np)


rating_file = '../data/' + 'book-ripple' + '/ratings_final'
rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)

a = []
for item in rating_np:
    if item[2] == 1:
        a.append(item)
rating_np_pos = np.array(a)
# np.save('../data/' + 'last-fm-small' + '/rating_pos' + '.npy', rating_np_pos)

# reading rating file
n_user = len(set(rating_np[:, 0]))
n_item = len(set(rating_np[:, 1]))

print('splitting dataset ...')

# train:test = 8:2
test_ratio = 0.2
n_ratings = rating_np_pos.shape[0]

test_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * test_ratio), replace=False)
train_indices = list(set(range(n_ratings)) - set(test_indices))

train_data = rating_np_pos[train_indices]
test_data = rating_np_pos[test_indices]
train_user_set = set(train_data[:, 0])
test_user_set = set(test_data[:, 1])
c=train_user_set & test_user_set
print('train user set is ', len(train_user_set))
print('test user set is ', len(test_user_set))
print('intersection is ', len(c))
print(1)

f = open('../data/' + 'book-ripple' + '/train_data1.pkl', 'wb')
pickle.dump(train_data, f)
fd = open('../data/' + 'book-ripple' + '/test_data1.pkl', 'wb')
pickle.dump(test_data, fd)



