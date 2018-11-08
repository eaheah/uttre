import os
import pickle
from tensorflow import keras
from random import shuffle

def create_partition(amount='all', datapath='/vagrant/imgs/training_data/training_data/aligned', split=(60, 20, 20)):
    directory = os.listdir(datapath)
    shuffle(directory)
    if amount != 'all':
        directory = directory[:amount]
    l = len(directory)
    train = int(l *split[0]/100)
    val = int(l * split[1]/100) + train
    test = int(l * split[2]/100) + val
    
    return {
        "train": directory[:train],
        "validation": directory[train:val],
        "test": directory[val:]
    }

def determine_attributes(prediction):
    label_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 
                   'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
                   'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
                   'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    label_dict = {i:name for i,name in enumerate(label_names)}
    reverse_label_dict = {name:i for i, name in label_dict.items()}
    
    related = [
        ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
        ['Straight_Hair', 'Wavy_Hair']
    ]

    reinforcement = {
        '5_o_Clock_Shadow': .1,
        'Goatee': .1,
        'Mustache': .1,
        'Sideburns': .1,
        'Wearing_Necktie': .1,
        'Heavy_Makeup': -.1,
        'Wearing_Earrings': -.1,
        'Wearing_Lipstick': -.1,
        'Wearing_Necklace': -.1,
    }
    
    threshold = .3
    male = 20
    
    predition = prediction.tolist()
    intermediary = []
    
    # high threshold
    for value in prediction:
        if value < threshold:
            intermediary.append(0)
        elif value > 1 - threshold:
            intermediary.append(1)
        else:
            intermediary.append(value)

    # reinforce gender  
    if intermediary[male] not in (1,0):
        for key in reinforcement:
            if intermediary[reverse_label_dict[key]] == 1:
                intermediary[male] += reinforcement[key]
    if intermediary[male] < threshold:
        intermediary[male] = 0
    elif intermediary[male] > 1 - threshold:
        intermediary[male] = 1
        
    # remove related if one is strong
    for d in related:
        print(d)
        if any([intermediary[reverse_label_dict[name]] == 1 for name in d]):
            for name in d:
                if name != 1:
                    del intermediary[reverse_label_dict[i]]
        
    results = {
        'sure': {'pos': [], 'neg': []},
        'unsure': {'pos': [], 'neg': []}
    }
    
    for i, value in enumerate(intermediary):
        name = label_dict[i]
        if value == 0:
            results['sure']['neg'].append(name)
        elif value == 1:
            results['sure']['pos'].append(name)
        elif value < .5:
            results['unsure']['neg'].append(name)
        else:
            results['unsure']['pos'].append(name)
    
    return results

def evaluate_model(model, data_generators, checkpoint_path, patience=20, period=5, workers=8, epochs=100, verbose=1):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=verbose, save_weights_only=True,
        period=period)
    
    history = model.fit_generator(generator=data_generators['training_generator'],
                        validation_data=data_generators['validation_generator'],
                        use_multiprocessing=True,
                        workers=workers,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=[early_stop, cp_callback])

    result = model.evaluate_generator(generator=data_generators['test_generator'], verbose=verbose)
    predictions = model.predict_generator(generator=data_generators['predition_generator'], verbose=verbose)
    return history, result, predictions

def md(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_pickled_partition(amount='all', padding='padding3'):
    path = f'data_partitions/{padding}/partition_{amount}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    
    data_partition_dir = 'data_partitions'
    for subdirectory in ('nopadding', 'padding1', 'padding2', 'padding3', 'padding4', 'padding5'):
        p = os.path.join(data_partition_dir, subdirectory)
        print(p)
        datapath = os.path.join('/vagrant/imgs/training_data2', subdirectory)
        print(datapath)

        for amount in (100, 1000, 100000, 'all'):
            partition = create_partition(amount=amount, datapath=datapath)
            out_path = os.path.join(p, 'partition_{}.pkl'.format(amount))
            with open(out_path, 'wb') as f:
                pickle.dump(partition, f)

    # with open('data_partitions/nopadding/partition_all.pkl', 'rb') as f:
    #     print(pickle.load(f).keys())

    # print()

    # with open('data_partitions/padding5/partition_all.pkl', 'rb') as f:
    #     print(pickle.load(f).keys())




